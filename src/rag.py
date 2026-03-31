import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import logging
from langchain_core.messages import HumanMessage, AIMessage
#Rerank相关
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
#BM25相关
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from typing import List, Set, Tuple, Any
from operator import itemgetter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.document_loaders import (
    PyPDFLoader,                # 专门处理 .pdf
    Docx2txtLoader,             # 专门处理 .docx
    UnstructuredWordDocumentLoader, # 处理 .doc
    TextLoader                  # 处理 .txt
)
from langchain_milvus import Milvus, BM25BuiltInFunction  # Milvus 向量库接口
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, utility, MilvusClient  # Milvus 连接管理

from database import MemoryManager
# 配置日志以查看生成的多个查询（可选）
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

DATA_PATH = "./data/" # 数据目录
INDEX_DIRECTORY = "./index"  # 索引目录
OUTPUT = "./workspace/ragoutput/answers_ensemble2.md" # 输出文件路径
QUERY = "请分别总结《The Semantic Architect: How FEAML Bridges Structured Data and LLMs for Multi-Label Tasks.pdf》的主要贡献。以及《12月4日开会总结.docx》的主要内容。" # 查询内容

# 定义后缀名与加载器的映射关系
# 格式: ".后缀": (Loader类, {初始化参数})
SUPPORTED_FILE_LOADERS = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
}

ZILLIZ_URI = "https://in03-6499fcae9264c18.serverless.gcp-us-west1.cloud.zilliz.com" # 替换为你复制的 Endpoint
ZILLIZ_TOKEN = "9aa32e1a5fb45bce452472d7f87f2906a504a3401887cb5e4e27492cb80443d2912d279133b3373ad7452a869b92e73c28b6b648" # 你的那个 *****8163 密钥
COLLECTION_NAME = "rag_collection_v2"  # Milvus 集合名称   


def format_docs(docs):
    """辅助函数：将检索到的文档格式化为带有页码的字符串"""
    formatted = []
    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", 0.0)
        content = doc.page_content.replace("\n", " ")
        # page = doc.metadata.get("page_label") or doc.metadata.get("page") or "未知来源"
        source = doc.metadata.get("source", "未知")
        filename = os.path.basename(source)
        # 打印调试，看看最终选了啥
        print(f"   > 最终入选 {i+1}: Score={score:.4f} | Source={source}")
        # 格式化输出
        formatted.append(f"[片段 {i+1} -  来源文件: {filename} - 相关性得分 {score:.2f}] : {content}")
    final_context = "\n".join(formatted)

    # 🔥 关键修复 2：用强烈的视觉符号打印最终喂给大模型的内容！
    # 如果这里打印出来的内容没有“开会细节”，说明 Docx2txtLoader 解析失败了，你需要检查 Word 文档。
    print("\n" + "▼"*40)
    print("【DEBUG: 真正被送进大模型的检索上下文】")
    print(final_context)
    print("▲"*40 + "\n")

    return final_context

def format_history(messages):
    """将历史消息格式化为可读字符串，便于拼接进提示词。"""
    if not messages:
        return "无"

    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "系统"
        formatted.append(f"{role}: {msg.content}")

    return "\n".join(formatted)

class UniqueRetriever(BaseRetriever):
    """
    一个包装器，用于对基础检索器的结果进行去重。
    去重标准：仅基于 page_content (正文内容)。
    """
    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. 调用基础检索器拿到所有文档（可能有重复）
        docs = self.base_retriever.invoke(query)
        
        # 2. 执行去重逻辑
        unique_docs = []
        seen_content = set()
        
        for doc in docs:
            # 去除首尾空格，防止微小格式差异
            content_signature = doc.page_content.strip()
            
            if content_signature not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_signature)
        
        # 可选：打印调试信息
        if len(docs) != len(unique_docs):
            print(f"🧹 触发去重: 原始 {len(docs)} -> 去重后 {len(unique_docs)}")
            
        return unique_docs

def all_files_in_dir(file_paths: list[str]) -> List[Document]:
        """加载指定目录下的所有支持格式的文件，返回 Document 列表"""
        documents = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()

            if ext in SUPPORTED_FILE_LOADERS:
                try:
                    loader_class, loader_args = SUPPORTED_FILE_LOADERS[ext]
                    loader = loader_class(file_path, **loader_args)
                    docs = loader.load()

                    for doc in docs:
                        #doc.page_content = f"文件名: {filename}\n内容: " + doc.page_content
                        doc.metadata["source"] = filename
                        doc.metadata["file_type"] = ext

                    documents.extend(docs)
                    print(f"已加载文件: {filename}，包含 {len(docs)} 页。")
                except Exception as e:
                    print(f"加载文件 {filename} 时出错: {e}")
            # 在每个文档前添加文件名信息
            else:
            # 遇到不支持的文件类型，跳过
            # print(f"   ⚠️ 跳过不支持的文件类型: {filename}")
                pass    
        if not documents:
            raise ValueError(f"在目录 {file_paths} 中未找到支持的文件。")
        return documents

def get_incremental_vector_store(embeddings):
    """
    增量更新向量库逻辑
    """
    existing_sources : Set[str] = set()

    connection_args = {
        "uri": ZILLIZ_URI,
        "token": ZILLIZ_TOKEN,
    }
    # 先建立注册连接，确保 MilvusClient 已经连接到 Zilliz 云端
    bootstrap_client = MilvusClient(**connection_args)
    # 找到 alias
    alias = bootstrap_client._using
    if not connections.has_connection(alias):
        connections.connect(alias=alias, **connection_args)

    vector_store = Milvus(
        # builtin_function=BM25BuiltInFunction(
        #     output_field_names=["sparse"],  # 确保 BM25 内置函数也能访问 source 字段
        # ),
        # vector_field = ["dense", "sparse"],  # 定义向量字段，dense 用于嵌入，sparse 用于 BM25
        connection_args=connection_args,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        auto_id=True,   # 👈 加上这个：Zilliz 云端强烈建议开启自动主键
        drop_old=False,   # 👈 加上这个：不删除旧索引，保持增量更新
    )
    print("vector_store 创建完成")
    print(vector_store)

    # 1.加载现有索引
    try:
        if vector_store.col is not None:
            print("开始 query existing sources")
            res = vector_store.col.query(
                expr="source != ''",  # 只查询有 source 的文档
                output_fields=["source"]
            )
            for item in res:
                if "source" in item:
                    existing_sources.add(item["source"])
            print(f"现有索引中已存在的文件: {existing_sources}")
    except Exception as e:
        print(f"加载现有索引时出错（如果是首次运行可能是正常的）: {e}")

    # 2.找出新文件
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    disk_files = [ f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    new_files_to_add = []

    for filename in disk_files:
        # 这里为了匹配元数据中的路径，通常 source 是文件的绝对或相对路径。
        # 请根据你 Document loader 中实际产生的 source 格式来比对。这里假设 source 存的是文件名或相对路径
        filepath = os.path.join(DATA_PATH, filename)
        
        # 核心逻辑：如果文件名不在 existing_sources 集合里，就是新文件
        # (注意：如果你的 existing_sources 里存的是完整路径，这里就要用 filepath 比较)
        if filename not in existing_sources:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_FILE_LOADERS:
                new_files_to_add.append(filepath)

    # 3.有新文件，就进行处理
    if new_files_to_add:
        print(f"发现 {len(new_files_to_add)} 个新文件需要添加到向量库。")
        new_docs = all_files_in_dir(new_files_to_add)

        if new_docs:
            # 切分
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            split_docs = text_splitter.split_documents(new_docs)
            print(f"✂️ 新文件切分为 {len(split_docs)} 个片段。")
            
            # 更新向量库
            vector_store.add_documents(split_docs)
        
        else:
            print("⚠️ 新文件加载后为空，未更新索引。")
    else:
        print("没有检测到新文件，直接使用现有索引")
        
    return vector_store


def main():
    openai_base_url = (os.getenv("OPENAI_API_BASE") or "").strip()
    model_kwargs = {
        "model": "deepseek-reasoner",  # 使用 DeepSeek Reasoner 模型
        "model_provider": "openai",  # 模型提供商
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.2,   # 较低温度，确保答案稳定
        "timeout": 120,
        "max_tokens": 3000,
    }
    if openai_base_url:
        model_kwargs["base_url"] = openai_base_url
    else:
        print("OPENAI_API_BASE is empty; using default OpenAI base_url.")

    model = init_chat_model(**model_kwargs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5" # 用 BGE向量嵌入模型
    )
    
    # 1.加载文件夹的pdf
    
    vector_store = get_incremental_vector_store(embeddings)

    # 2.反向提取，切分文档
    # docs = list(vector_store.docstore._dict.values())
    

    # # 2.1 去掉明显噪声段（致谢/参考文献等）
    # noise_markers = ("References", "Acknowledgments")
    # docs = [
    #     d for d in docs
    #     if not any(m in d.page_content for m in noise_markers)
    # ]

    # 3.生成向量
    # A.创建 FAISS 向量库
    # vector_store = FAISS.from_documents(docs, embeddings)
    # print("向量数量:", vector_store.index.ntotal)  
    
    # 4.创建检索器
    # A. 构建混合检索器
    # 基础向量检索器
    # faiss_retriever = vector_store.as_retriever(
    #     search_type="mmr", 
    #     search_kwargs={"k":10, "fetch_k":30}
    # )

    # 加上效果更差
    # multiquery_retriever = MultiQueryRetriever.from_llm(
    #         retriever=faiss_retriever,
    #         llm=model,
    # )
    
    # BM25检索器
    # bm25_retriever = BM25Retriever.from_documents(
    #     documents=docs
    # )
    # bm25_retriever.k = 10 # 设置返回的文档数量  

    # # 组合向量检索器和BM25检索器
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[faiss_retriever, bm25_retriever],
    #     weights=[0.8, 0.2]  # 可以根据需要调整权重
    # )
    hybrid_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,        # 最终返回的文档数量
            # "ranker_type": "weighted",
            # "ranker_params": {"weights": [0.8, 0.2]}  # 向量检索和BM25的权重
        }
    )
    # B. 使用 UniqueRetriever 包裹 EnsembleRetriever
    # 流程：Ensemble -> UniqueRetriever(去重) -> FlashRank
    deduplicated_retriever = UniqueRetriever(base_retriever=hybrid_retriever)

    # C. FlashRank (专业裁判) - 这是一个 Cross-Encoder
    # 它会自动下载约 90MB 的模型文件到本地
    try:
        compressor = FlashrankRerank(top_n=5)
        # D. 组装最终检索器：MultiQuery -> 去重(内部处理) -> FlashRank重排 -> Top 5
        # ContextualCompressionRetriever 会自动处理文档的打分和筛选
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=deduplicated_retriever,
        )
    except Exception as e:
        print(f"FlashRank init failed, skipping rerank: {e}")
        compression_retriever = deduplicated_retriever

    # 5 Prompt 模板
    rerank_prompt = ChatPromptTemplate.from_template(
        """
    你是一位专家助手。请基于以下经过【相关性重排序】的片段回答问题。
    片段已按重要性排序（分数越高越重要）。只用上下文回答，缺失就说不知道

    【上下文片段】：
    {context}

    【历史对话】：
    {history}

    【用户问题】：
    {question}
    """
    )

    # 6.创建文档处理链，｜代表把前一步喂给下一步
    rag_chain = (
        {
            "context": itemgetter("query") | compression_retriever | format_docs,
            "question": itemgetter("query"),
            "history": itemgetter("history") | RunnableLambda(format_history),
        }
        | rerank_prompt
        | model
        | StrOutputParser()
    )

    # 初始化分层记忆管理器
    memory_manager = MemoryManager(session_id="Jack01", summary_llm=model,retain_n_turns=3)

    while True:
        user_input = input("请输入你的问题（或输入 'exit' 退出）：")
        if user_input.lower() == "exit":
            print("退出程序。")
            break

        # 1.获取当前上下文（摘要 + 近期对话）
        history_messages = memory_manager.get_context_messages()

        print("\nAI: ", end="", flush=True)
        response_content = ""

        # 2.执行 RAG 问答链，传入上下文和用户查询
        for chunk in rag_chain.stream({
            "query": user_input,
            "history": history_messages,
        }):
            print(chunk, end="", flush=True)
            response_content += chunk

        # 3.保存对话上下文（用户提问 + 模型回答）
        memory_manager.save_context(human_text=user_input, ai_text=response_content)
        
        # 4.输出结果并保存到文件
        output_path = OUTPUT
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response_content)
        print(f"问答完成，答案已保存到 {output_path}")
        print(response_content)

if __name__ == "__main__":
    main()
