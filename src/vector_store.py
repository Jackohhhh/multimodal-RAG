"""
向量库工厂：统一管理 FAISS / Milvus 的构建与加载。
"""
import json
import os
import shutil
from typing import List

from langchain_core.documents import Document


def _normalize_metadata_for_faiss(documents: List[Document]) -> List[Document]:
    """FAISS 可直接保存 list metadata，无需额外序列化。"""
    return documents


def _normalize_metadata_for_milvus(documents: List[Document]) -> List[Document]:
    """Milvus 对 list metadata 支持较差，写入前转为 JSON 字符串。"""
    normalized = []
    for doc in documents:
        cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
        imgs = cloned.metadata.get("related_images")
        if isinstance(imgs, list):
            cloned.metadata["related_images"] = json.dumps(imgs, ensure_ascii=False)
        normalized.append(cloned)
    return normalized


def build_vector_store(documents: List[Document], embeddings, vector_store_config: dict):
    """
    根据配置构建向量库并持久化。
    支持:
    - type: faiss
    - type: milvus
    """
    store_type = vector_store_config.get("type", "faiss").lower()

    if store_type == "faiss":
        from langchain_community.vectorstores import FAISS

        index_path = vector_store_config["index_path"]
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        vector_store = FAISS.from_documents(
            _normalize_metadata_for_faiss(documents),
            embeddings,
        )
        vector_store.save_local(index_path)
        return vector_store

    if store_type == "milvus":
        from langchain_milvus import Milvus
        from pymilvus import connections, utility
        from pymilvus import MilvusClient

        conn_args = vector_store_config["connection_args"]
        collection_name = vector_store_config["collection_name"]

        bootstrap_client = MilvusClient(**conn_args)
        alias = bootstrap_client._using
        if not connections.has_connection(alias):
            connections.connect(alias=alias, **conn_args)

        if utility.has_collection(collection_name, using=alias):
            utility.drop_collection(collection_name, using=alias)

        vector_store = Milvus(
            connection_args=conn_args,
            collection_name=collection_name,
            embedding_function=embeddings,
            auto_id=True,
            drop_old=False,
        )
        vector_store.add_documents(_normalize_metadata_for_milvus(documents))
        return vector_store

    raise ValueError(f"不支持的向量库类型: {store_type}")


def load_vector_store(embeddings, vector_store_config: dict):
    """
    根据配置加载已有向量库。
    """
    store_type = vector_store_config.get("type", "faiss").lower()

    if store_type == "faiss":
        from langchain_community.vectorstores import FAISS

        index_path = vector_store_config["index_path"]
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS 索引不存在: {index_path}。请先运行 `python3 run.py preprocess` 构建索引。"
            )
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if store_type == "milvus":
        from langchain_milvus import Milvus
        from pymilvus import connections, MilvusClient

        conn_args = vector_store_config["connection_args"]
        bootstrap_client = MilvusClient(**conn_args)
        alias = bootstrap_client._using
        if not connections.has_connection(alias):
            connections.connect(alias=alias, **conn_args)

        return Milvus(
            connection_args=conn_args,
            collection_name=vector_store_config["collection_name"],
            embedding_function=embeddings,
            auto_id=True,
            drop_old=False,
        )

    raise ValueError(f"不支持的向量库类型: {store_type}")
