"""
多模态融合检索：文本检索 + 图片ID自动带出。
图片ID已在预处理阶段写入每个 chunk 的 metadata["related_images"]，
检索时直接跟着文本片段一起返回，无需额外查找。
"""
import json
from typing import List

from langchain_core.documents import Document


class MultimodalRetriever:
    """
    检索器封装：执行文本检索后，每个命中的 Document
    自带 metadata["related_images"] = ["drill0_04", "drill0_05", ...]。
    """

    def __init__(self, text_retriever):
        self.text_retriever = text_retriever

    def retrieve(self, query: str) -> List[Document]:
        docs = self.text_retriever.invoke(query)
        for doc in docs:
            imgs = doc.metadata.get("related_images")
            if isinstance(imgs, str):
                try:
                    doc.metadata["related_images"] = json.loads(imgs)
                except (json.JSONDecodeError, TypeError):
                    doc.metadata["related_images"] = []
        return docs
