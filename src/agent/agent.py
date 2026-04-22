"""
多模态客服智能体：主 Agent 入口。
整合思维链拆解、RAG 检索、幻觉抑制。
输出格式："回答文本(含<PIC>)", [图片ID列表]
"""
import os
import re
import yaml
from typing import List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import format_docs, postprocess_answer, log_retrieved_docs
from src.retrieval.text_retriever import build_text_retriever
from src.retrieval.manual_source_rules import (
    filter_documents_by_manual_source,
    is_mainly_english_query,
)
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.agent.chain_of_thought import decompose_question, merge_answers
from src.agent.hallucination import check_grounding, LOW_CONFIDENCE_DISCLAIMER
from src.agent.multimodal_input import build_multimodal_query
from src.vector_store import load_vector_store


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return yaml.safe_load(os.path.expandvars(raw))


def load_prompts(prompts_path: str = "configs/prompts.yaml") -> dict:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CustomerServiceAgent:
    """多模态客服智能体"""

    GENERIC_CUSTOMER_SERVICE_PATTERNS = (
        r"退款",
        r"退货",
        r"换货",
        r"取消订单",
        r"订单.*取消",
        r"发票",
        r"物流",
        r"快递",
        r"揽收",
        r"签收",
        r"投诉",
        r"补发",
        r"补寄",
        r"少发",
        r"漏发",
        r"运费",
        r"赔偿",
        r"售后",
        r"到货",
        r"到账",
    )

    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.prompts = prompts

        self.llm = self._init_llm(config["llm"])
        self.multimodal_llm = self._init_llm(config.get("multimodal_llm", config["llm"]))
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model_name"]
        )
        self.vector_store = load_vector_store(
            self.embeddings,
            config["vector_store"],
        )

        retrieval_cfg = config["retrieval"]
        data_cfg = config.get("data") or {}
        chunks_path = data_cfg.get("chunks_output")
        if chunks_path:
            chunks_path = os.path.expandvars(chunks_path)
        fr_cache = retrieval_cfg.get("flashrank_cache_dir")
        if fr_cache:
            fr_cache = os.path.expandvars(str(fr_cache))
        self.text_retriever = build_text_retriever(
            self.vector_store,
            search_k=retrieval_cfg["search_k"],
            rerank_top_n=retrieval_cfg["rerank_top_n"],
            use_rerank=retrieval_cfg["use_rerank"],
            use_hybrid_bm25=bool(retrieval_cfg.get("use_hybrid_bm25", False)),
            vector_top_k=int(retrieval_cfg.get("vector_top_k", 4)),
            bm25_top_k=int(retrieval_cfg.get("bm25_top_k", 4)),
            chunks_jsonl_path=chunks_path,
            flashrank_model=retrieval_cfg.get("flashrank_model"),
            flashrank_cache_dir=fr_cache,
        )
        self.rag_relevance_threshold = float(
            retrieval_cfg.get("rag_relevance_threshold", 0.40)
        )
        cap = retrieval_cfg.get("rag_max_context_documents")
        self.rag_max_context_documents = int(cap) if cap is not None else None
        self.english_rag_only_no_customer_service_llm = bool(
            retrieval_cfg.get("english_rag_only_no_customer_service_llm", True)
        )

        self.multimodal_retriever = MultimodalRetriever(
            text_retriever=self.text_retriever,
        )

        self.rag_prompt = ChatPromptTemplate.from_template(
            prompts["rag_prompt"]
        )
        fallback_prompt = prompts.get("fallback_customer_service_prompt") or (
            "你现在扮演商家在线客服。请直接回答用户问题，不要提及知识库、说明书、检索或模型。"
            "\n\n【用户问题】：\n{question}\n\n【客服回答】："
        )
        self.customer_service_fallback_prompt = ChatPromptTemplate.from_template(
            fallback_prompt
        )
        self.decompose_prompt_tpl = prompts.get("decompose_prompt")
        self.merge_prompt_tpl = prompts.get("merge_prompt")
        rw_tpl = prompts.get("retrieval_query_rewrite_prompt")
        self.retrieval_query_rewrite_prompt = (
            ChatPromptTemplate.from_template(rw_tpl) if rw_tpl else None
        )

    def _init_llm(self, llm_config: dict):
        provider = (llm_config.get("provider") or "openai").strip().lower()
        kwargs = {
            "model": llm_config["model"],
            "model_provider": llm_config["provider"],
            "temperature": llm_config.get("temperature", 0.2),
            "timeout": llm_config.get("timeout", 120),
            "max_tokens": llm_config.get("max_tokens", 3000),
        }
        if provider == "google_genai":
            # Gemini Developer API：用 GOOGLE_API_KEY / GEMINI_API_KEY；勿沿用 OpenAI 网关 base_url
            api_key = (
                llm_config.get("api_key")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
            if api_key:
                kwargs["api_key"] = api_key
            base_url = (llm_config.get("base_url") or "").strip()
            if base_url:
                kwargs["base_url"] = base_url
        else:
            api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            base_url = (
                llm_config.get("base_url") or os.getenv("OPENAI_API_BASE", "").strip()
            )
            if base_url:
                kwargs["base_url"] = base_url
        return init_chat_model(**kwargs)

    def answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        enable_cot: bool = False,
        enable_hallucination_check: bool = False,
    ) -> Tuple[str, List[str]]:
        """
        核心问答接口。
        返回: (含<PIC>的回答文本, 图片ID列表)
        """
        query = build_multimodal_query(question, image_path, self.multimodal_llm)

        if self._should_use_customer_service_directly(
            query
        ) and self._may_use_customer_service_llm(question):
            fallback_answer = self._fallback_customer_service_answer(question)
            text_with_pic, image_ids = postprocess_answer(fallback_answer)
            return text_with_pic, image_ids

        if enable_cot:
            sub_questions = decompose_question(
                query, self.llm, prompt_template=self.decompose_prompt_tpl
            )
        else:
            sub_questions = [query]

        sub_answers = self._collect_sub_answers(
            question, sub_questions, enable_hallucination_check
        )
        raw_answer = merge_answers(
            question, sub_answers, self.llm,
            prompt_template=self.merge_prompt_tpl,
        )

        if (
            is_mainly_english_query(question)
            and self.retrieval_query_rewrite_prompt is not None
            and self._is_exact_rag_miss_message(raw_answer)
        ):
            rewritten = self._rewrite_retrieval_query(question)
            if rewritten and not self._same_query_for_retrieval(rewritten, question):
                sub_answers_retry = self._collect_sub_answers(
                    question,
                    [query],
                    enable_hallucination_check,
                    retrieval_queries=[rewritten],
                )
                raw_answer = merge_answers(
                    question,
                    sub_answers_retry,
                    self.llm,
                    prompt_template=self.merge_prompt_tpl,
                )

        text_with_pic, image_ids = postprocess_answer(raw_answer)
        return text_with_pic, image_ids

    def _collect_sub_answers(
        self,
        question: str,
        sub_questions: List[str],
        enable_hallucination_check: bool,
        retrieval_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """逐条子问题 RAG；可选与 sub_questions 等长的 retrieval_queries 仅用于向量/BM25 检索。"""
        if retrieval_queries is not None and len(retrieval_queries) != len(
            sub_questions
        ):
            raise ValueError(
                "retrieval_queries 必须与 sub_questions 等长。"
            )
        sub_answers: List[str] = []
        for i, sub_q in enumerate(sub_questions):
            rq = (
                retrieval_queries[i]
                if retrieval_queries is not None
                else sub_q
            )
            docs = self.multimodal_retriever.retrieve(rq)
            pre_filter_n = len(docs)
            docs = self._filter_docs_above_rag_threshold(docs)
            docs = filter_documents_by_manual_source(question, docs)
            docs = self._limit_rag_context_documents(docs)
            log_retrieved_docs(
                rq,
                docs,
                pre_filter_hit_count=pre_filter_n,
                rag_relevance_threshold=self.rag_relevance_threshold,
            )
            if not docs:
                sub_answers.append(
                    self._no_rag_docs_answer(question, sub_q)
                )
                continue

            context = format_docs(docs)

            if not context.strip():
                sub_answers.append(
                    self._no_rag_docs_answer(question, sub_q)
                )
                continue

            response = (
                self.rag_prompt
                | self.llm
                | StrOutputParser()
            ).invoke({
                "context": context,
                "question": sub_q,
            })

            if self._needs_customer_service_fallback(response):
                sub_answers.append(
                    self._rag_no_hit_answer(question, sub_q, response)
                )
                continue

            if enable_hallucination_check:
                grounding = check_grounding(context, response, self.llm)
                if grounding == "HALLUCINATION":
                    sub_answers.append(
                        self._hallucination_fallback_answer(question, sub_q)
                    )
                    continue
                if grounding == "PARTIAL":
                    response = response + LOW_CONFIDENCE_DISCLAIMER

            sub_answers.append(response)
        return sub_answers

    def _rewrite_retrieval_query(self, question: str) -> str:
        """将用户问句改写为更适合检索的英文短查询（单行）。"""
        chain = self.retrieval_query_rewrite_prompt | self.llm | StrOutputParser()
        out = chain.invoke({"question": question}).strip()
        if not out:
            return ""
        first = out.splitlines()[0].strip()
        first = first.strip("\"'“”‘’")
        return first

    @staticmethod
    def _is_exact_rag_miss_message(text: str) -> bool:
        """合并后的 RAG 输出是否仅为提示词规定的无命中句（触发英文 query 重写）。"""
        normalized = re.sub(r"\s+", "", (text or "").strip())
        return normalized == "未找到相关信息"

    @staticmethod
    def _same_query_for_retrieval(a: str, b: str) -> bool:
        """改写后与原文若等价则跳过重检索，避免无效二次调用。"""
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        return norm(a) == norm(b)

    @staticmethod
    def _retrieval_score_for_threshold(doc: Document) -> Optional[float]:
        """精排后优先用粗排分（retrieval_score）做阈值，避免归一化精排分与 0.4 阈值不对齐。"""
        m = doc.metadata or {}
        rs = m.get("retrieval_score")
        if rs is not None:
            try:
                return float(rs)
            except (TypeError, ValueError):
                pass
        rv = m.get("relevance_score")
        if rv is not None:
            try:
                return float(rv)
            except (TypeError, ValueError):
                pass
        return None

    def _filter_docs_above_rag_threshold(self, docs: List[Document]) -> List[Document]:
        """仅保留粗排分严格大于 rag_relevance_threshold 的片段（有 retrieval_score 时以它为准）。"""
        thr = self.rag_relevance_threshold
        kept: List[Document] = []
        for doc in docs:
            raw = self._retrieval_score_for_threshold(doc)
            if raw is None:
                continue
            if raw > thr:
                kept.append(doc)
        return kept

    def _limit_rag_context_documents(self, docs: List[Document]) -> List[Document]:
        """
        截取进入 prompt 的条数；保持检索器返回顺序（精排序），不再按分数重排，
        避免把「向量高分但非最贴题」的段重新顶到最前。
        """
        cap = self.rag_max_context_documents
        if cap is None or cap <= 0:
            return docs
        return docs[:cap]

    def _may_use_customer_service_llm(self, question: str) -> bool:
        """英文为主且开启 english_rag_only 时，禁止走客服兜底大模型。"""
        if not self.english_rag_only_no_customer_service_llm:
            return True
        return not is_mainly_english_query(question or "")

    def _english_rag_no_evidence_answer(self, sub_question: str) -> str:
        """无说明书片段时，仅用 RAG 主提示词 + 主 LLM 生成英文式无依据答复（不调用客服提示词）。"""
        ctx = (
            "[System: No manual excerpts were retrieved. Answer in English only. "
            "Briefly state that the product documentation available here does not "
            "contain information to answer this question. Do not invent specifications "
            "or procedures.]"
        )
        return (
            self.rag_prompt
            | self.llm
            | StrOutputParser()
        ).invoke({"context": ctx, "question": sub_question}).strip()

    def _no_rag_docs_answer(self, question: str, sub_question: str) -> str:
        if self._may_use_customer_service_llm(question):
            return self._fallback_customer_service_answer(sub_question)
        return self._english_rag_no_evidence_answer(sub_question)

    def _rag_no_hit_answer(self, question: str, sub_question: str, response: str) -> str:
        """RAG 返回无答案信号时：中文可走客服；英文保留 RAG 输出或再经 RAG 主模型补一句。"""
        if self._may_use_customer_service_llm(question):
            return self._fallback_customer_service_answer(sub_question)
        text = (response or "").strip()
        if text:
            return text
        return self._english_rag_no_evidence_answer(sub_question)

    def _hallucination_fallback_answer(self, question: str, sub_question: str) -> str:
        if self._may_use_customer_service_llm(question):
            return self._fallback_customer_service_answer(sub_question)
        return self._english_rag_no_evidence_answer(sub_question)

    def _fallback_customer_service_answer(self, question: str) -> str:
        """说明书缺少依据时，切换到商家客服口径回答。"""
        return (
            self.customer_service_fallback_prompt
            | self.llm
            | StrOutputParser()
        ).invoke({"question": question}).strip()

    @staticmethod
    def _needs_customer_service_fallback(answer: str) -> bool:
        """识别 RAG 提示词返回的无答案信号。"""
        normalized = re.sub(r"\s+", "", answer or "")
        fallback_signals = (
            "未找到相关信息",
            "没有相关信息",
            "无法找到相关信息",
            "无法从上下文中找到",
        )
        return (not normalized) or any(signal in normalized for signal in fallback_signals)

    @classmethod
    def _should_use_customer_service_directly(cls, question: str) -> bool:
        """通用售后类问题优先走客服兜底，不进入说明书 RAG。"""
        normalized = re.sub(r"\s+", "", question or "")
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in cls.GENERIC_CUSTOMER_SERVICE_PATTERNS)
