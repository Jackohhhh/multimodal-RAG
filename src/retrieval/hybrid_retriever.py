"""
向量 Top-K + BM25 Top-K 合并，写入 relevance_score 供阈值过滤。
"""
from __future__ import annotations

import re
from typing import Any, Callable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever


def bm25_tokenize_zh(text: str) -> List[str]:
    if not text or not text.strip():
        return [" "]
    parts = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    return parts if parts else [" "]


def collect_corpus_from_faiss(vector_store: Any) -> List[Document]:
    docstore = getattr(vector_store, "docstore", None)
    if docstore is None:
        return []
    dct = getattr(docstore, "_dict", None)
    if not isinstance(dct, dict):
        return []
    return list(dct.values())


def _vector_search_with_scores(vectorstore: Any, query: str, k: int) -> List[Document]:
    pairs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    out: List[Document] = []
    for doc, sim in pairs:
        meta = dict(doc.metadata)
        meta["relevance_score"] = float(sim)
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out


class HybridVectorBm25Retriever(BaseRetriever):
    vectorstore: Any
    bm25_retriever: Any
    vector_k: int = 8
    bm25_top_k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        vec_docs = _vector_search_with_scores(self.vectorstore, query, self.vector_k)
        bm25_all = self.bm25_retriever.invoke(query)
        vec_sigs = {d.page_content.strip() for d in vec_docs}
        bm25_unique: List[Document] = []
        n = max(self.bm25_top_k, 1)
        for doc in bm25_all:
            if len(bm25_unique) >= self.bm25_top_k:
                break
            sig = doc.page_content.strip()
            if sig in vec_sigs:
                continue
            r = len(bm25_unique) + 1
            aux = (n - r + 1) / n * 0.55
            meta = dict(doc.metadata)
            meta["relevance_score"] = float(aux)
            bm25_unique.append(
                Document(page_content=doc.page_content, metadata=meta)
            )

        return list(vec_docs) + bm25_unique


def build_bm25_retriever(
    corpus: List[Document],
    k: int,
    preprocess_func: Optional[Callable[[str], List[str]]] = None,
) -> BM25Retriever:
    fn = preprocess_func or bm25_tokenize_zh
    return BM25Retriever.from_documents(
        corpus,
        k=max(k, 32),
        preprocess_func=fn,
    )


def resolve_bm25_corpus(
    vector_store: Any,
    chunks_jsonl_path: Optional[str] = None,
) -> List[Document]:
    import os

    corpus = collect_corpus_from_faiss(vector_store)
    if corpus:
        return corpus
    if chunks_jsonl_path and os.path.isfile(chunks_jsonl_path):
        from src.preprocess.chunk_text import load_chunks_from_jsonl

        return load_chunks_from_jsonl(chunks_jsonl_path)
    return []
