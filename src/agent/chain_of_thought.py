"""
思维链拆解：将用户一次提问中的多个子问题拆分，逐一检索作答后汇总。
"""
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def decompose_question(question: str, llm, prompt_template: str = None) -> List[str]:
    """将复杂问题拆解为子问题列表"""
    if prompt_template is None:
        prompt_template = (
            "请分析用户的提问，判断其中是否包含多个独立子问题。\n"
            "如果包含多个子问题，请将它们逐条列出（每行一个，用数字编号）。\n"
            "如果只有一个问题，直接输出该问题即可。\n\n"
            "【用户提问】：\n{question}\n\n【拆分结果】："
        )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
    sub_questions = []
    for line in lines:
        cleaned = line.lstrip("0123456789.、).） ").strip()
        if cleaned:
            sub_questions.append(cleaned)

    return sub_questions if sub_questions else [question]


def merge_answers(
    original_question: str,
    sub_answers: List[str],
    llm,
    prompt_template: str = None,
) -> str:
    """将多个子问题的回答合并为最终回答"""
    if len(sub_answers) == 1:
        return sub_answers[0]

    if prompt_template is None:
        prompt_template = (
            "将以下子回答合并为一份简洁的最终回答。\n"
            "规则：保留所有 [IMG:xxx] 标记，不要使用 Markdown 格式。\n\n"
            "【用户原始提问】：\n{original_question}\n\n"
            "【各子问题的回答】：\n{sub_answers}\n\n"
            "【合并后的最终回答】："
        )

    formatted = "\n\n".join(
        f"子问题 {i+1} 的回答：\n{ans}" for i, ans in enumerate(sub_answers)
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "original_question": original_question,
        "sub_answers": formatted,
    })
