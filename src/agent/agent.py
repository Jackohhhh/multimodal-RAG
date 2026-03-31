"""
多模态客服智能体：主 Agent 入口。
整合思维链拆解、RAG 检索、幻觉抑制。
输出格式："回答文本(含<PIC>)", [图片ID列表]
"""
import os
import yaml
from typing import Optional, Tuple, List

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import format_docs, postprocess_answer
from src.retrieval.text_retriever import build_text_retriever
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.agent.chain_of_thought import decompose_question, merge_answers
from src.agent.hallucination import apply_hallucination_filter
from src.agent.multimodal_input import build_multimodal_query


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return yaml.safe_load(os.path.expandvars(raw))


def load_prompts(prompts_path: str = "configs/prompts.yaml") -> dict:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CustomerServiceAgent:
    """多模态客服智能体"""

    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.prompts = prompts

        self.llm = self._init_llm(config["llm"])
        self.multimodal_llm = self._init_llm(config.get("multimodal_llm", config["llm"]))
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model_name"]
        )

        from langchain_milvus import Milvus
        from pymilvus import connections, MilvusClient

        conn_args = config["vector_store"]["connection_args"]
        bootstrap_client = MilvusClient(**conn_args)
        alias = bootstrap_client._using
        if not connections.has_connection(alias):
            connections.connect(alias=alias, **conn_args)

        self.vector_store = Milvus(
            connection_args=conn_args,
            collection_name=config["vector_store"]["collection_name"],
            embedding_function=self.embeddings,
            auto_id=True,
            drop_old=False,
        )

        self.text_retriever = build_text_retriever(
            self.vector_store,
            search_k=config["retrieval"]["search_k"],
            rerank_top_n=config["retrieval"]["rerank_top_n"],
            use_rerank=config["retrieval"]["use_rerank"],
        )

        self.multimodal_retriever = MultimodalRetriever(
            text_retriever=self.text_retriever,
        )

        self.rag_prompt = ChatPromptTemplate.from_template(
            prompts["rag_prompt"]
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

        if enable_cot:
            sub_questions = decompose_question(
                query, self.llm, prompt_template=self.decompose_prompt_tpl
            )
        else:
            sub_questions = [query]

        sub_answers = []

        for sub_q in sub_questions:
            docs = self.multimodal_retriever.retrieve(sub_q)
            context = format_docs(docs)

            response = (
                self.rag_prompt
                | self.llm
                | StrOutputParser()
            ).invoke({
                "context": context,
                "question": sub_q,
            })

            filtered = apply_hallucination_filter(
                context=context,
                answer=response,
                llm=self.llm,
                enable_check=enable_hallucination_check,
            )
            sub_answers.append(filtered)

        raw_answer = merge_answers(
            question, sub_answers, self.llm,
            prompt_template=self.merge_prompt_tpl,
        )

        text_with_pic, image_ids = postprocess_answer(raw_answer)
        return text_with_pic, image_ids
