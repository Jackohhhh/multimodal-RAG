"""
构建图片向量库：将图片 caption 索引到向量库中，支持以文搜图。
"""
import os
import json
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_captions_as_documents(captions_path: str) -> List[Document]:
    """将 caption JSON 加载为 Document 列表，每个 caption 作为一个文档"""
    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    documents = []
    for cap in captions:
        caption_text = cap.get("caption", "").strip()
        if not caption_text:
            continue
        doc = Document(
            page_content=caption_text,
            metadata={
                "source": cap.get("image_filename", ""),
                "image_path": cap.get("image_path", ""),
                "doc_type": "image_caption",
            },
        )
        documents.append(doc)

    print(f"从 caption 文件加载了 {len(documents)} 个图片描述文档")
    return documents


def build_image_vector_store(
    captions_path: str,
    embeddings,
    vector_store_class,
    **vector_store_kwargs,
):
    """
    构建图片 caption 向量库。
    将图片描述文本向量化并索引，用于"以文搜图"检索。
    """
    docs = load_captions_as_documents(captions_path)
    if not docs:
        print("没有可用的图片 caption，跳过图片索引构建。")
        return None

    vector_store = vector_store_class.from_documents(
        docs, embeddings, **vector_store_kwargs
    )
    print(f"图片 caption 向量库构建完成，共 {len(docs)} 条")
    return vector_store
