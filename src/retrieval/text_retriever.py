"""
文本检索器：向量检索 + 去重 + FlashRank Rerank。
从现有 rag.py 中的检索管线拆分而来。
"""
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


class UniqueRetriever(BaseRetriever):
    """对基础检索器的结果按 page_content 去重"""

    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        unique_docs = []
        seen_content = set()
        for doc in docs:
            content_signature = doc.page_content.strip()
            if content_signature not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_signature)
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
    向量检索 → 去重 → FlashRank Rerank → Top N
    """
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": search_k}
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
