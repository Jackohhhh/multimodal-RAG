"""
幻觉抑制：对检索结果进行置信度评估，拒绝低置信度回答。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个事实核查助手。请判断以下【回答】是否完全基于【上下文】中的信息。

评判标准：
1. 回答中的每个关键事实都能在上下文中找到依据 → "GROUNDED"
2. 回答中有部分内容无法从上下文中验证 → "PARTIAL"
3. 回答中的关键内容在上下文中没有依据或与上下文矛盾 → "HALLUCINATION"

【上下文】：
{context}

【回答】：
{answer}

请只输出一个词：GROUNDED、PARTIAL 或 HALLUCINATION"""
)

NO_CONTEXT_ANSWER = (
    "抱歉，根据现有知识库中的资料，我无法找到与您问题相关的信息。"
    "请尝试换一种方式描述您的问题，或联系人工客服获取帮助。"
)

LOW_CONFIDENCE_DISCLAIMER = (
    "\n\n⚠️ 提示：以上回答可能不够完整，建议结合说明书原文确认。"
)


def check_grounding(context: str, answer: str, llm) -> str:
    """
    检查回答是否基于上下文。
    返回: "GROUNDED" / "PARTIAL" / "HALLUCINATION"
    """
    chain = VERIFY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer}).strip().upper()
    for label in ("GROUNDED", "PARTIAL", "HALLUCINATION"):
        if label in result:
            return label
    return "PARTIAL"


def apply_hallucination_filter(
    context: str,
    answer: str,
    llm,
    enable_check: bool = True,
) -> str:
    """
    对回答做幻觉过滤：
    - 无上下文时直接拒答
    - HALLUCINATION 时拒答
    - PARTIAL 时追加提示
    - GROUNDED 时原样返回
    """
    if not context or not context.strip():
        return NO_CONTEXT_ANSWER

    if not enable_check:
        return answer

    grounding = check_grounding(context, answer, llm)
    if grounding == "HALLUCINATION":
        return NO_CONTEXT_ANSWER
    elif grounding == "PARTIAL":
        return answer + LOW_CONFIDENCE_DISCLAIMER
    return answer
