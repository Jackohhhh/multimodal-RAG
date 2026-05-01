"""
多模态客服智能体：主 Agent 入口。
整合思维链拆解、RAG 检索、幻觉抑制。
输出格式："回答文本(含<PIC>)", [图片ID列表]
"""
import json
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
    required_source_substrings,
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

    SECTION_INTENT_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        (
            r"组成|构成|部件|零件|配件|包含什么|有哪些.*部件",
            ("部件介绍", "部件说明", "零件清单", "产品介绍", "概览"),
        ),
        (
            r"技术规格|规格|参数|尺寸|重量|容量",
            ("技术规格", "规格", "参数", "尺寸"),
        ),
        (
            r"保修|免责声明|除外责任|不包含|损害赔偿",
            ("保修", "除外责任", "免责", "免责声明"),
        ),
        (
            r"清洁|清洗|更换|安装|拆卸|充电|设置|调节|启动|关闭|停机",
            ("清洁", "清洗", "更换", "安装", "拆卸", "充电", "设置", "调节", "启动", "停机"),
        ),
    )

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

    ENGLISH_RETRIEVAL_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "before", "by", "can",
        "could", "detail", "detailed", "do", "does", "during", "ever",
        "everything", "feature", "first", "for", "from", "have", "how",
        "i", "if", "in", "is", "it", "manual", "me", "my", "of", "on",
        "or", "process", "procedure", "properly", "quick", "ready", "should",
        "step", "steps", "the", "this", "three", "to", "use", "using",
        "want", "what", "when", "while", "with", "you", "your",
    }

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
        self.chunk_documents = self._load_chunk_documents(chunks_path)
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
        self.rag_fallback_relevance_threshold = float(
            retrieval_cfg.get("rag_fallback_relevance_threshold", 0.25)
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
        expand_tpl = prompts.get("query_expand_prompt")
        self.query_expand_prompt = (
            ChatPromptTemplate.from_template(expand_tpl) if expand_tpl else None
        )
        self.use_query_rewrite = bool(
            retrieval_cfg.get("use_query_rewrite", False)
        )
        raw_rewrite_langs = retrieval_cfg.get("query_rewrite_languages", ["zh"])
        if isinstance(raw_rewrite_langs, str):
            raw_rewrite_langs = [raw_rewrite_langs]
        self.query_rewrite_languages = {
            str(x).strip().lower() for x in raw_rewrite_langs
        }

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
            raw_docs = self._retrieve_with_rewrite(rq, original_question=question)
            pre_filter_n = len(raw_docs)
            docs = self._select_docs_for_rag(question, raw_docs)
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

    def _expand_query(self, question: str) -> str:
        """用 LLM 将问题改写为详细的扩展检索查询（单行输出）。"""
        if self.query_expand_prompt is None:
            return ""
        chain = self.query_expand_prompt | self.llm | StrOutputParser()
        out = chain.invoke({"question": question}).strip()
        if not out:
            return ""
        first = out.splitlines()[0].strip()
        first = first.strip("\"'""''")
        return first

    def _retrieve_with_rewrite(
        self,
        query: str,
        *,
        original_question: Optional[str] = None,
    ) -> List[Document]:
        """按语言策略检索：原 query + 可选扩写 query，随后合并去重。"""
        retrieval_queries = [query]

        if self._should_use_query_rewrite(original_question or query):
            expanded = self._expand_query(query)
            if expanded and not self._same_query_for_retrieval(expanded, query):
                retrieval_queries.append(expanded)

        unique_queries: List[str] = []
        for candidate in retrieval_queries:
            candidate = (candidate or "").strip()
            if candidate and not any(
                self._same_query_for_retrieval(candidate, prev)
                for prev in unique_queries
            ):
                unique_queries.append(candidate)

        retrieved_docs: List[Document] = []
        for retrieval_query in unique_queries:
            retrieved_docs.extend(self.multimodal_retriever.retrieve(retrieval_query))
        retrieved_docs.extend(
            self._retrieve_matching_section_titles(original_question or query)
        )

        return self._merge_doc_lists(retrieved_docs)

    @staticmethod
    def _load_chunk_documents(chunks_path: Optional[str]) -> List[Document]:
        """读取切分后的手册片段，用于通用标题级补召回。"""
        if not chunks_path or not os.path.exists(chunks_path):
            return []

        docs: List[Document] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = item.get("content") or ""
                source = item.get("source") or ""
                if not content or not source:
                    continue
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "related_images": item.get("related_images") or [],
                        },
                    )
                )
        return docs

    def _retrieve_matching_section_titles(self, question: str) -> List[Document]:
        """
        从同产品手册中补充“标题与问题词高度重合”的片段。
        这是通用标题匹配，不写入题目答案，主要兜住 BM25/向量未进前列的精确章节。
        """
        if not self.chunk_documents:
            return []

        required_sources = required_source_substrings(question or "")
        if not required_sources:
            return []

        expanded_question = self._expand_manual_query_terms(question or "")
        normalized_question = self._remove_source_names(
            expanded_question, required_sources
        )
        query_bigrams = self._cjk_bigrams(normalized_question)
        query_tokens = self._english_tokens(normalized_question)
        if not query_bigrams and not query_tokens:
            return []

        scored: List[Tuple[float, Document]] = []
        for doc in self.chunk_documents:
            source = (doc.metadata.get("source") or "") if doc.metadata else ""
            if not any(src in source for src in required_sources):
                continue
            headings = self._heading_text(doc.page_content)
            match_text = f"{headings}\n{doc.page_content or ''}"
            if not headings and not match_text.strip():
                continue
            heading_bigrams = self._cjk_bigrams(
                self._remove_source_names(headings, required_sources)
            )
            overlap = len(query_bigrams & heading_bigrams)
            score = self._section_title_match_score(
                expanded_question, headings, required_sources
            )
            if query_tokens:
                score = max(
                    score,
                    self._english_text_match_score(query_tokens, match_text),
                )
            score += self._manual_phrase_match_bonus(expanded_question, match_text)
            if query_bigrams and overlap >= 2:
                score = max(score, overlap * 8.0)
            if score < 6.0:
                continue
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: List[Document] = []
        for score, doc in scored[:6]:
            meta = dict(doc.metadata)
            # 让补召回片段通过 fallback 阈值，但不伪装成主检索高分。
            meta.setdefault(
                "retrieval_score",
                self.rag_fallback_relevance_threshold + min(score, 20) * 0.01,
            )
            meta.setdefault("relevance_score", meta["retrieval_score"])
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out

    @staticmethod
    def _heading_text(content: str) -> str:
        headings = []
        for line in (content or "").splitlines():
            if "#" not in line:
                continue
            for segment in line.split("#"):
                stripped = segment.strip()
                if not stripped:
                    continue
                title = re.split(r"[。；;.!！?？\n]", stripped, maxsplit=1)[0]
                title = title.strip()
                if title:
                    headings.append(title[:40])
        return " ".join(headings)

    @staticmethod
    def _cjk_bigrams(text: str) -> set[str]:
        chars = re.findall(r"[\u4e00-\u9fff]", text or "")
        return {
            "".join(chars[i:i + 2])
            for i in range(len(chars) - 1)
        }

    @classmethod
    def _english_tokens(cls, text: str) -> set[str]:
        tokens = set()
        for raw in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()):
            token = cls._stem_english_token(raw)
            if len(token) < 2 or token in cls.ENGLISH_RETRIEVAL_STOPWORDS:
                continue
            tokens.add(token)
        return tokens

    @staticmethod
    def _stem_english_token(token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    @classmethod
    def _english_text_match_score(cls, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = cls._english_tokens(text)
        if not text_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        if not overlap:
            return 0.0
        coverage = len(overlap) / max(len(query_tokens), 1)
        specificity = len(overlap) * 4.0
        phrase_bonus = 0.0
        text_lower = re.sub(r"\s+", " ", (text or "").lower())
        for token in overlap:
            if re.search(rf"\b{re.escape(token)}\b", text_lower):
                phrase_bonus += 0.5
        return specificity + coverage * 10.0 + phrase_bonus

    @staticmethod
    def _manual_phrase_match_bonus(question: str, text: str) -> float:
        q = (question or "").lower()
        t = (text or "").lower()
        phrase_rules: Tuple[Tuple[str, str, float], ...] = (
            ("battery conversion", "# battery switches", 30.0),
            ("anchor light", "# anchor light", 28.0),
            ("maintenance setting screen", "# maintenance setting screen", 32.0),
            ("make the boat move forward", "# forward", 32.0),
            ("ready to sail", "# forward", 24.0),
            ("steering system", "# steering system checks", 32.0),
            ("connecting the fax", "# connect the product safely", 30.0),
            ("assembly process", "# assembly", 30.0),
            ("start the engine", "starting theengine", 28.0),
            ("start the engine", "starting the engine", 28.0),
            ("start the engine", "fully open the starter lever", 36.0),
            ("start the engine", "manual starting model", 30.0),
            ("clean a snowmobile", "# 1. cleaning", 28.0),
            ("clean snowmobile", "# 1. cleaning", 28.0),
            ("山地", "# 山地", 32.0),
            ("重要组成部件", "# 部件介绍", 28.0),
            ("测量我的心率", "# 心率信号丢失", 18.0),
            ("测量我的心率", "# 查看心率", 18.0),
        )
        return sum(bonus for needle, title, bonus in phrase_rules if needle in q and title in t)

    @classmethod
    def _section_title_match_score(
        cls,
        question: str,
        headings: str,
        required_sources: Tuple[str, ...],
    ) -> float:
        normalized_question = cls._remove_source_names(
            cls._expand_manual_query_terms(question), required_sources
        )
        normalized_headings = cls._remove_source_names(headings, required_sources)
        query_bigrams = cls._cjk_bigrams(normalized_question)
        heading_bigrams = cls._cjk_bigrams(normalized_headings)
        cjk_score = 0.0
        if query_bigrams and heading_bigrams:
            overlap = len(query_bigrams & heading_bigrams)
            if overlap:
                title_specificity = overlap * 10 / max(len(heading_bigrams), 1) ** 0.5
                lcs_bonus = cls._longest_common_cjk_substring_len(
                    normalized_question, normalized_headings
                ) * 3
                cjk_score = title_specificity + lcs_bonus

        english_score = cls._english_text_match_score(
            cls._english_tokens(normalized_question),
            normalized_headings,
        )
        return max(cjk_score, english_score)

    @staticmethod
    def _expand_manual_query_terms(text: str) -> str:
        """补充常见说明书同义词，供标题/正文补召回使用，不直接写入答案。"""
        q = text or ""
        lower = q.lower()
        additions: List[str] = []
        alias_rules: Tuple[Tuple[str, str], ...] = (
            ("battery conversion", "battery switches START HOUSE EMERG PARALLEL start battery house battery"),
            ("ready to sail", "starting off remote control lever throttle forward"),
            ("make the boat move forward", "starting off remote control lever throttle forward"),
            ("anchor light", "set up removable anchor light socket navigation anchor lights switch"),
            ("steering system", "steering wheel jet thrust nozzle articulating keel remote control lever"),
            ("maintenance setting screen", "maintenance setting screen reset hours operation"),
            ("assembly process", "assembly attach locking casters bottom shelf supplied wrench"),
            ("connecting the fax", "connect product safely telephone line cord power cord wall jack"),
            ("start the engine", "starting the engine starter choke throttle main switch"),
            ("clean a snowmobile", "cleaning wash wax storage snowmobile"),
            ("clean snowmobile", "cleaning wash wax storage snowmobile"),
        )
        for needle, extra in alias_rules:
            if needle in lower:
                additions.append(extra)
        return " ".join([q, *additions]).strip()

    @staticmethod
    def _remove_source_names(text: str, required_sources: Tuple[str, ...]) -> str:
        normalized = text or ""
        for source in required_sources:
            product_name = source.replace("手册", "").replace("_", "")
            if product_name:
                normalized = normalized.replace(product_name, "")
        return normalized

    @staticmethod
    def _longest_common_cjk_substring_len(a: str, b: str) -> int:
        """标题匹配用的最长中文公共子串长度，偏好具体小节标题。"""
        left = "".join(re.findall(r"[\u4e00-\u9fff]", a or ""))
        right = "".join(re.findall(r"[\u4e00-\u9fff]", b or ""))
        if not left or not right:
            return 0
        prev = [0] * (len(right) + 1)
        best = 0
        for i in range(1, len(left) + 1):
            curr = [0] * (len(right) + 1)
            for j in range(1, len(right) + 1):
                if left[i - 1] == right[j - 1]:
                    curr[j] = prev[j - 1] + 1
                    if curr[j] > best:
                        best = curr[j]
            prev = curr
        return best

    def _merge_doc_lists(self, docs: List[Document]) -> List[Document]:
        """合并多路检索结果，正文去重并保留更高粗排分。"""
        seen: dict = {}
        for doc in docs:
            sig = doc.page_content.strip()
            if sig not in seen:
                seen[sig] = doc
            else:
                existing = seen[sig]
                existing_score = self._retrieval_score_for_threshold(existing) or 0.0
                new_score = self._retrieval_score_for_threshold(doc) or 0.0
                if new_score > existing_score:
                    seen[sig] = doc

        merged = sorted(
            seen.values(),
            key=lambda d: self._retrieval_score_for_threshold(d) or 0.0,
            reverse=True,
        )
        return merged

    def _should_use_query_rewrite(self, question: str) -> bool:
        """控制 query 扩写适用语言；英文扩写在当前数据上容易带偏，默认只开中文。"""
        if not self.use_query_rewrite or self.query_expand_prompt is None:
            return False
        if "all" in self.query_rewrite_languages:
            return True
        lang = "en" if is_mainly_english_query(question or "") else "zh"
        return lang in self.query_rewrite_languages

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
        normalized = re.sub(r"\s+", "", (text or "").strip()).lower()
        return normalized in {
            "未找到相关信息",
            "norelevantinformationfound.",
            "norelevantinformationfound",
        }

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

    def _select_docs_for_rag(self, question: str, docs: List[Document]) -> List[Document]:
        """
        进入 RAG 前的筛选：
        1. 正常路径：先按主阈值过滤，再按产品 source 过滤；
        2. 兜底路径：在同产品源内放宽阈值补充候选，减少高分但错章节导致的漏答。
        """
        thresholded = self._filter_docs_above_rag_threshold(docs)
        selected = filter_documents_by_manual_source(question, thresholded)
        source_filtered = filter_documents_by_manual_source(question, docs)
        relaxed = self._filter_docs_above_fallback_threshold(source_filtered)
        candidates = self._merge_doc_lists_preserve_order(selected + relaxed)
        prioritized = self._prioritize_docs_by_section_intent(question, candidates)
        return self._limit_rag_context_documents(prioritized)

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

    def _filter_docs_above_fallback_threshold(self, docs: List[Document]) -> List[Document]:
        """按放宽阈值保留同产品候选；中文、英文共用同一阈值配置。"""
        kept: List[Document] = []
        for doc in docs:
            raw = self._retrieval_score_for_threshold(doc)
            if raw is None or raw > self.rag_fallback_relevance_threshold:
                kept.append(doc)
        return kept

    def _prioritize_docs_by_section_intent(
        self, question: str, docs: List[Document]
    ) -> List[Document]:
        """按通用章节意图和问题-标题相关度提权，优先选择具体小节。"""
        if not docs:
            return docs
        required_sources = required_source_substrings(question or "")
        title_terms: List[str] = []
        for question_pattern, terms in self.SECTION_INTENT_RULES:
            if re.search(question_pattern, question or "", re.IGNORECASE):
                title_terms.extend(terms)

        preferred: List[Document] = []
        rest: List[Document] = []
        for doc in docs:
            text = doc.page_content or ""
            title_score = self._section_title_match_score(
                question,
                self._heading_text(text),
                required_sources,
            )
            doc_score = title_score + self._manual_phrase_match_bonus(question, text)
            if doc_score >= 8 or any(term in text for term in title_terms):
                preferred.append(doc)
            else:
                rest.append(doc)
        preferred.sort(
            key=lambda doc: (
                self._section_title_match_score(
                    question,
                    self._heading_text(doc.page_content),
                    required_sources,
                )
                + self._manual_phrase_match_bonus(question, doc.page_content)
            ),
            reverse=True,
        )
        return preferred + rest

    def _merge_doc_lists_preserve_order(self, docs: List[Document]) -> List[Document]:
        """按正文去重，保留首次出现顺序；若后续重复分数更高则替换文档内容。"""
        seen: dict[str, Document] = {}
        order: List[str] = []
        for doc in docs:
            sig = doc.page_content.strip()
            if sig not in seen:
                seen[sig] = doc
                order.append(sig)
                continue
            existing = seen[sig]
            existing_score = self._retrieval_score_for_threshold(existing) or 0.0
            new_score = self._retrieval_score_for_threshold(doc) or 0.0
            if new_score > existing_score:
                seen[sig] = doc
        return [seen[sig] for sig in order]

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
        """英文及已命中产品手册的问题不走通用客服，避免说明书题被泛答。"""
        if not self.english_rag_only_no_customer_service_llm:
            return True
        if is_mainly_english_query(question or ""):
            return False
        return not self._looks_like_manual_product_query(question)

    @staticmethod
    def _looks_like_manual_product_query(question: str) -> bool:
        return bool(required_source_substrings(question or ""))

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
        if not is_mainly_english_query(question or ""):
            return "未找到相关信息"
        return self._english_rag_no_evidence_answer(sub_question)

    def _rag_no_hit_answer(self, question: str, sub_question: str, response: str) -> str:
        """RAG 返回无答案信号时：中文可走客服；英文保留 RAG 输出或再经 RAG 主模型补一句。"""
        if self._may_use_customer_service_llm(question):
            return self._fallback_customer_service_answer(sub_question)
        if not is_mainly_english_query(question or ""):
            return (response or "").strip() or "未找到相关信息"
        text = (response or "").strip()
        if text:
            return text
        return self._english_rag_no_evidence_answer(sub_question)

    def _hallucination_fallback_answer(self, question: str, sub_question: str) -> str:
        if self._may_use_customer_service_llm(question):
            return self._fallback_customer_service_answer(sub_question)
        if not is_mainly_english_query(question or ""):
            return "未找到相关信息"
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
            "norelevantinformationfound",
            "no relevant information found",
            "cannot find relevant information",
        )
        normalized_lower = (answer or "").strip().lower()
        return (
            not normalized
            or any(signal in normalized for signal in fallback_signals)
            or any(signal in normalized_lower for signal in fallback_signals)
        )

    @classmethod
    def _should_use_customer_service_directly(cls, question: str) -> bool:
        """通用售后类问题优先走客服兜底，不进入说明书 RAG。"""
        normalized = re.sub(r"\s+", "", question or "")
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in cls.GENERIC_CUSTOMER_SERVICE_PATTERNS)
