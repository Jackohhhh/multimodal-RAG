"""
文本检索器：向量检索 + 去重 + FlashRank Rerank。
从现有 rag.py 中的检索管线拆分而来。
"""
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


class VectorStoreScoredRetriever(BaseRetriever):
    """
    使用 similarity_search_with_relevance_scores，将 [0,1] 相似度写入
    metadata[\"relevance_score\"]。（FAISS / Milvus 均支持。）
    """

    vectorstore: Any
    search_k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        pairs = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.search_k
        )
        out: List[Document] = []
        for doc, sim in pairs:
            meta = dict(doc.metadata)
            meta["relevance_score"] = float(sim)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out


class UniqueRetriever(BaseRetriever):
    """对基础检索器的结果按 page_content 去重；同正文保留 relevance_score 较高的一条"""

    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        best_by_sig: dict[str, Document] = {}
        for doc in docs:
            content_signature = doc.page_content.strip()
            sc = float(doc.metadata.get("relevance_score", 0.0))
            if content_signature not in best_by_sig:
                best_by_sig[content_signature] = doc
            else:
                prev_sc = float(
                    best_by_sig[content_signature].metadata.get("relevance_score", 0.0)
                )
                if sc > prev_sc:
                    best_by_sig[content_signature] = doc
        unique_docs = list(best_by_sig.values())
        if len(docs) != len(unique_docs):
            print(f"触发去重: 原始 {len(docs)} -> 去重后 {len(unique_docs)}")
        return unique_docs


def build_text_retriever(
    vector_store,
    search_k: int = 10,
    rerank_top_n: int = 5,
    use_rerank: bool = True,
):
    """
    构建完整的文本检索管线：
    向量检索（带 relevance_score）→ 去重 → FlashRank Rerank → Top N
    """
    base_retriever = VectorStoreScoredRetriever(
        vectorstore=vector_store,
        search_k=search_k,
    )

    deduplicated = UniqueRetriever(base_retriever=base_retriever)

    if not use_rerank:
        return deduplicated

    try:
        compressor = FlashrankRerank(top_n=rerank_top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=deduplicated,
        )
    except Exception as e:
        print(f"FlashRank 初始化失败，跳过 Rerank: {e}")
        return deduplicated
