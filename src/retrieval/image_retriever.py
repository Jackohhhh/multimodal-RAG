"""
图片ID提取工具：从检索结果的 metadata 中收集关联的图片ID。
在新的 <PIC> 占位符方案下，图片ID已预存在每个 chunk 的 metadata 中，
本模块仅做提取和去重。
"""
from typing import List
from langchain_core.documents import Document


def get_image_ids_from_docs(docs: List[Document]) -> List[str]:
    """从检索结果中收集所有关联的图片ID（去重保序）"""
    seen = set()
    result = []
    for doc in docs:
        for img_id in doc.metadata.get("related_images", []):
            if img_id not in seen:
                result.append(img_id)
                seen.add(img_id)
    return result
