"""
构建文本向量库：增量更新 Milvus/Zilliz 向量索引。
"""
import os
from typing import Set

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.preprocess.parse_manual import load_documents, SUPPORTED_FILE_LOADERS
from src.vector_store import build_vector_store


def build_text_vector_store(
    data_path: str,
    embeddings,
    vector_store_config: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
):
    """
    构建文本向量库。
    扫描 data_path 目录，切分后写入目标向量库。
    """
    existing_sources: Set[str] = set()

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    disk_files = [
        f for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
    ]

    new_files = []
    for filename in disk_files:
        if filename in existing_sources:
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_FILE_LOADERS:
            new_files.append(os.path.join(data_path, filename))

    if new_files:
        print(f"发现 {len(new_files)} 个文件需要写入向量库。")
        new_docs = load_documents(new_files)
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            split_docs = text_splitter.split_documents(new_docs)
            print(f"文件切分为 {len(split_docs)} 个片段。")
            return build_vector_store(split_docs, embeddings, vector_store_config)
        else:
            print("文件加载后为空，未更新索引。")
    else:
        print("没有检测到可用文件，未构建索引。")
    return None
