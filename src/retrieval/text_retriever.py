"""
文本检索：可选「向量 + BM25」混合粗排 → 去重 → FlashRank 精排。

精排后写入：
- metadata["retrieval_score"]：粗排分数（向量/BM25），供阈值过滤；
- metadata["relevance_score"]：精排在当前批次内归一化到 [0,1]，便于日志与兼容旧逻辑；
- metadata["rerank_score"]：精排原始分。

避免使用 LangChain 自带 FlashrankRerank 将 **r["meta"] 后展开盖住精排分** 的问题。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

from pydantic import ConfigDict, model_validator
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever


class VectorStoreScoredRetriever(BaseRetriever):
    """similarity_search_with_relevance_scores，分数写入 metadata[\"relevance_score\"]。"""

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
    """按正文去重；保留分数较高的一条，输出顺序与「首次命中」粗排顺序一致。"""

    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        best_by_sig: dict[str, Document] = {}
        order: List[str] = []
        for doc in docs:
            content_signature = doc.page_content.strip()
            sc = float(doc.metadata.get("relevance_score", 0.0))
            if content_signature not in best_by_sig:
                best_by_sig[content_signature] = doc
                order.append(content_signature)
            else:
                prev = best_by_sig[content_signature]
                prev_sc = float(prev.metadata.get("relevance_score", 0.0))
                if sc > prev_sc:
                    best_by_sig[content_signature] = doc
        unique_docs = [best_by_sig[s] for s in order]
        if len(docs) != len(unique_docs):
            print(f"触发去重: 原始 {len(docs)} -> 去重后 {len(unique_docs)}")
        return unique_docs


class MetadataPreservingFlashrankRerank(BaseDocumentCompressor):
    """FlashRank 精排，并保留粗排分在 retrieval_score，避免阈值与日志失真。"""

    top_n: int = 8
    # 默认用较小 ONNX，首次下载更快；大模型见 configs/config.yaml 说明
    model_name: str = "ms-marco-MiniLM-L-12-v2"
    cache_dir: Optional[str] = None
    client: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _ensure_client(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("client") is not None:
            return data
        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "请安装 flashrank：`pip install flashrank`"
            ) from e
        mn = data.get("model_name", "ms-marco-MiniLM-L-12-v2")
        raw_cache = data.get("cache_dir")
        if raw_cache:
            cache_dir = os.path.abspath(os.path.expandvars(str(raw_cache)))
        else:
            cache_dir = str(Path(__file__).resolve().parents[2] / "models" / "flashrank")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        data["client"] = Ranker(model_name=mn, cache_dir=cache_dir)
        return data

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": d.page_content, "meta": dict(d.metadata)}
            for i, d in enumerate(documents)
        ]
        if not passages:
            return []
        resp = self.client.rerank(RerankRequest(query=query, passages=passages))
        resp = resp[: self.top_n]
        scores = [float(r["score"]) for r in resp]
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi > lo else 1.0

        out: List[Document] = []
        for r in resp:
            base_meta = dict(r["meta"])
            raw_vec = base_meta.get("relevance_score")
            try:
                retrieval = float(raw_vec) if raw_vec is not None else None
            except (TypeError, ValueError):
                retrieval = None
            rs = float(r["score"])
            norm = (rs - lo) / span if span else 1.0
            meta = {**base_meta, "rerank_score": rs, "relevance_score": float(norm)}
            if retrieval is not None:
                meta["retrieval_score"] = retrieval
            out.append(Document(page_content=r["text"], metadata=meta))
        return out


def build_text_retriever(
    vector_store,
    search_k: int = 10,
    rerank_top_n: int = 5,
    use_rerank: bool = True,
    *,
    use_hybrid_bm25: bool = False,
    vector_top_k: int = 4,
    bm25_top_k: int = 4,
    chunks_jsonl_path: Optional[str] = None,
    flashrank_model: Optional[str] = None,
    flashrank_cache_dir: Optional[str] = None,
):
    """
    粗排：可选 Hybrid（vector_top_k + bm25_top_k），否则纯向量 search_k 条。
    去重后交给 FlashRank 精排，输出 top rerank_top_n；精排顺序即最终顺序。
    """
    if use_hybrid_bm25:
        from src.retrieval.hybrid_retriever import (
            HybridVectorBm25Retriever,
            resolve_bm25_corpus,
            build_bm25_retriever,
        )

        corpus = resolve_bm25_corpus(vector_store, chunks_jsonl_path)
        if corpus:
            bm25_fetch_k = max(32, bm25_top_k * 8, vector_top_k + bm25_top_k * 4)
            bm25 = build_bm25_retriever(corpus, k=bm25_fetch_k)
            # 精排前多拉向量候选，避免「真相关段」未进粗排池（不只针对某一类产品）
            vector_k = max(vector_top_k, search_k)
            inner = HybridVectorBm25Retriever(
                vectorstore=vector_store,
                bm25_retriever=bm25,
                vector_k=vector_k,
                bm25_top_k=bm25_top_k,
            )
        else:
            print(
                "警告：未找到 BM25 语料，回退纯向量；可调大 search_k 以增加精排候选。"
            )
            inner = VectorStoreScoredRetriever(
                vectorstore=vector_store,
                search_k=search_k,
            )
    else:
        inner = VectorStoreScoredRetriever(
            vectorstore=vector_store,
            search_k=search_k,
        )

    deduplicated = UniqueRetriever(base_retriever=inner)

    if not use_rerank:
        print("Rerank 未启用：配置 retrieval.use_rerank 为 false。")
        return deduplicated

    fr_name = flashrank_model or "ms-marco-MiniLM-L-12-v2"
    try:
        compressor = MetadataPreservingFlashrankRerank(
            top_n=rerank_top_n,
            model_name=fr_name,
            cache_dir=flashrank_cache_dir,
        )
        print(f"Rerank 已启用：FlashRank 模型「{fr_name}」。")
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=deduplicated,
        )
    except Exception as e:
        zip_url = (
            "https://huggingface.co/prithivida/flashrank/resolve/main/"
            f"{fr_name}.zip"
        )
        print(
            "FlashRank 初始化失败 → Rerank 未启用，当前仅「粗排 + 去重」（无精排）。\n"
            f"原因: {e}\n"
            f"说明: 首次运行需从 HuggingFace 下载模型；若 SSL/网络不通，请用浏览器或 wget 下载上述 zip，"
            f"解压到 cache 目录下，使存在文件夹「{fr_name}/」（内含 .onnx 与 tokenizer 等文件）。\n"
            f"zip 地址: {zip_url}"
        )
        return deduplicated
