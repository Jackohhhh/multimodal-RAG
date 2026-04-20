"""
多模态客服智能体：主 Agent 入口。
整合思维链拆解、RAG 检索、幻觉抑制。
输出格式："回答文本(含<PIC>)", [图片ID列表]
"""
import os
import re
import yaml
from typing import Optional, Tuple, List

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import format_docs, postprocess_answer, log_retrieved_docs
from src.retrieval.text_retriever import build_text_retriever
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
        self.text_retriever = build_text_retriever(
            self.vector_store,
            search_k=retrieval_cfg["search_k"],
            rerank_top_n=retrieval_cfg["rerank_top_n"],
            use_rerank=retrieval_cfg["use_rerank"],
        )
        self.rag_relevance_threshold = float(
            retrieval_cfg.get("rag_relevance_threshold", 0.40)
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

    def _init_llm(self, llm_config: dict):
        kwargs = {
            "model": llm_config["model"],
            "model_provider": llm_config["provider"],
            "temperature": llm_config.get("temperature", 0.2),
            "timeout": llm_config.get("timeout", 120),
            "max_tokens": llm_config.get("max_tokens", 3000),
        }
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        base_url = llm_config.get("base_url") or os.getenv("OPENAI_API_BASE", "").strip()
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

        if self._should_use_customer_service_directly(query):
            fallback_answer = self._fallback_customer_service_answer(question)
            text_with_pic, image_ids = postprocess_answer(fallback_answer)
            return text_with_pic, image_ids

        if enable_cot:
            sub_questions = decompose_question(
                query, self.llm, prompt_template=self.decompose_prompt_tpl
            )
        else:
            sub_questions = [query]

        sub_answers = []

        for sub_q in sub_questions:
            docs = self.multimodal_retriever.retrieve(sub_q)
            pre_filter_n = len(docs)
            docs = self._filter_docs_above_rag_threshold(docs)
            log_retrieved_docs(
                sub_q,
                docs,
                pre_filter_hit_count=pre_filter_n,
                rag_relevance_threshold=self.rag_relevance_threshold,
            )
            if not docs:
                sub_answers.append(self._fallback_customer_service_answer(sub_q))
                continue

            context = format_docs(docs)

            if not context.strip():
                sub_answers.append(self._fallback_customer_service_answer(sub_q))
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
                sub_answers.append(self._fallback_customer_service_answer(sub_q))
                continue

            if enable_hallucination_check:
                grounding = check_grounding(context, response, self.llm)
                if grounding == "HALLUCINATION":
                    sub_answers.append(self._fallback_customer_service_answer(sub_q))
                    continue
                if grounding == "PARTIAL":
                    response = response + LOW_CONFIDENCE_DISCLAIMER

            sub_answers.append(response)

        raw_answer = merge_answers(
            question, sub_answers, self.llm,
            prompt_template=self.merge_prompt_tpl,
        )

        text_with_pic, image_ids = postprocess_answer(raw_answer)
        return text_with_pic, image_ids

    def _filter_docs_above_rag_threshold(self, docs: List[Document]) -> List[Document]:
        """仅保留 relevance_score 严格大于 rag_relevance_threshold 的片段供 RAG 使用。"""
        thr = self.rag_relevance_threshold
        kept: List[Document] = []
        for doc in docs:
            raw = doc.metadata.get("relevance_score")
            if raw is None:
                continue
            try:
                if float(raw) > thr:
                    kept.append(doc)
            except (TypeError, ValueError):
                continue
        return kept

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
