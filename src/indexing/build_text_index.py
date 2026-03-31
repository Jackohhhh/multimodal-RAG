"""
构建文本向量库：增量更新 Milvus/Zilliz 向量索引。
"""
import os
from typing import Set, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections, MilvusClient

from src.preprocess.parse_manual import load_documents, SUPPORTED_FILE_LOADERS


def build_text_vector_store(
    data_path: str,
    embeddings,
    connection_args: dict,
    collection_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    drop_old: bool = False,
):
    """
    增量构建文本向量库。
    扫描 data_path 目录，只将新文件的切片加入向量库。
    """
    existing_sources: Set[str] = set()

    bootstrap_client = MilvusClient(**connection_args)
    alias = bootstrap_client._using
    if not connections.has_connection(alias):
        connections.connect(alias=alias, **connection_args)

    vector_store = Milvus(
        connection_args=connection_args,
        collection_name=collection_name,
        embedding_function=embeddings,
        auto_id=True,
        drop_old=drop_old,
    )

    try:
        if vector_store.col is not None:
            res = vector_store.col.query(
                expr="source != ''",
                output_fields=["source"],
            )
            for item in res:
                if "source" in item:
                    existing_sources.add(item["source"])
            print(f"现有索引中已存在的文件: {existing_sources}")
    except Exception as e:
        print(f"加载现有索引时出错（首次运行可能是正常的）: {e}")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    disk_files = [
        f for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
    ]

    new_files = []
    for filename in disk_files:
        if filename not in existing_sources:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_FILE_LOADERS:
                new_files.append(os.path.join(data_path, filename))

    if new_files:
        print(f"发现 {len(new_files)} 个新文件需要添加到向量库。")
        new_docs = load_documents(new_files)
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            split_docs = text_splitter.split_documents(new_docs)
            print(f"新文件切分为 {len(split_docs)} 个片段。")
            vector_store.add_documents(split_docs)
        else:
            print("新文件加载后为空，未更新索引。")
    else:
        print("没有检测到新文件，直接使用现有索引。")

    return vector_store
