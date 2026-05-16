"""
向量 Top-K + BM25 Top-K 使用 RRF（Reciprocal Rank Fusion）融合排序。
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    rrf_k: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        vec_docs = _vector_search_with_scores(self.vectorstore, query, self.vector_k)
        bm25_docs = self.bm25_retriever.invoke(query)[: self.bm25_top_k]

        fused: Dict[str, Tuple[Document, float, float]] = {}
        for rank, doc in enumerate(vec_docs, start=1):
            self._accumulate_rrf_score(
                fused,
                doc,
                rank=rank,
                source_name="vector",
                original_score=float(doc.metadata.get("relevance_score", 0.0)),
            )

        for rank, doc in enumerate(bm25_docs, start=1):
            self._accumulate_rrf_score(
                fused,
                doc,
                rank=rank,
                source_name="bm25",
                original_score=0.0,
            )

        ranked = sorted(
            fused.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        if not ranked:
            return []

        max_rrf = ranked[0][1] or 1.0
        out: List[Document] = []
        for doc, rrf_score, best_original_score in ranked:
            meta = dict(doc.metadata)
            meta["rrf_score"] = float(rrf_score)
            meta["retrieval_score"] = float(rrf_score / max_rrf)
            meta["relevance_score"] = float(rrf_score / max_rrf)
            if best_original_score:
                meta["original_relevance_score"] = float(best_original_score)
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out

    def _accumulate_rrf_score(
        self,
        fused: Dict[str, Tuple[Document, float, float]],
        doc: Document,
        *,
        rank: int,
        source_name: str,
        original_score: float,
    ) -> None:
        sig = doc.page_content.strip()
        score = 1.0 / (self.rrf_k + rank)
        if sig in fused:
            existing_doc, existing_rrf, best_original = fused[sig]
            meta = dict(existing_doc.metadata)
            sources = set(meta.get("retrieval_sources", []))
            sources.add(source_name)
            meta["retrieval_sources"] = sorted(sources)
            if original_score > best_original:
                meta.update(dict(doc.metadata))
                existing_doc = Document(
                    page_content=doc.page_content,
                    metadata=meta,
                )
                best_original = original_score
            fused[sig] = (existing_doc, existing_rrf + score, best_original)
            return

        meta = dict(doc.metadata)
        meta["retrieval_sources"] = [source_name]
        fused[sig] = (
            Document(page_content=doc.page_content, metadata=meta),
            score,
            original_score,
        )


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
