"""公共工具函数"""
import re
from typing import List, Tuple
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """
    将检索到的文档格式化为上下文字符串。
    文本中的 [IMG:xxx] 标记保留，大模型可据此引用对应图片。
    """
    formatted = []

    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", 0.0)
        source = doc.metadata.get("source", "未知")
        content = doc.page_content

        formatted.append(
            f"[片段 {i+1} - 来源: {source} - 得分 {score:.2f}]:\n{content}"
        )

    return "\n\n".join(formatted)


def collect_image_ids(docs: List[Document]) -> List[str]:
    """从检索结果中收集所有关联的图片ID（去重保序）"""
    seen = set()
    result = []
    for doc in docs:
        for img_id in doc.metadata.get("related_images", []):
            if img_id not in seen:
                result.append(img_id)
                seen.add(img_id)
    return result


def postprocess_answer(answer: str) -> Tuple[str, List[str]]:
    """
    后处理大模型回答：
    1. 提取回答中所有 [IMG:xxx] 标记，按出现顺序收集图片ID
    2. 将 [IMG:xxx] 替换回 <PIC>
    返回: (含<PIC>的文本, 图片ID列表)
    """
    image_ids = re.findall(r'\[IMG:([^\]]+)\]', answer)
    text_with_pic = re.sub(r'\n?\[IMG:[^\]]+\]\n?', '<PIC>', answer)
    return text_with_pic, image_ids
