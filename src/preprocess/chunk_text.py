"""
文本切分：将标记后的文本切分为片段，每个片段自动携带其包含的图片ID。
按 # 标题优先切分，保证语义完整性。
"""
import os
import re
import json
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _split_by_headers(text: str) -> List[str]:
    """
    按 # 标题行切分文本，每个标题及其下属内容作为一个独立段落。
    标题行前的内容（如果有）作为第一段。
    """
    parts = re.split(r'(?=\n# )', text)
    sections = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            sections.append(stripped)
    return sections


def chunk_marked_text(
    marked_text: str,
    source: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    切分含 [IMG:xxx] 标记的文本。

    策略：先按 # 标题硬切，保证每个章节独立；
    仅对超过 chunk_size 的长章节进一步切分。
    每个 chunk 的 metadata 自动包含该 chunk 内出现的图片ID列表。
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    sections = _split_by_headers(marked_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )

    all_chunks: List[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            all_chunks.append(section)
        else:
            sub_chunks = text_splitter.split_text(section)
            all_chunks.extend(sub_chunks)

    documents = []
    for i, chunk_text in enumerate(all_chunks):
        image_ids = re.findall(r'\[IMG:([^\]]+)\]', chunk_text)

        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": source,
                "chunk_id": i,
                "related_images": image_ids,
            },
        )
        documents.append(doc)

    total_images = sum(len(d.metadata["related_images"]) for d in documents)
    print(
        f"切分完成: {len(documents)} 个片段，"
        f"其中 {sum(1 for d in documents if d.metadata['related_images'])} 个含图片，"
        f"共关联 {total_images} 张图"
    )
    return documents


def save_chunks_to_jsonl(chunks: List[Document], output_path: str):
    """将切分结果保存为 JSONL"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.metadata.get("chunk_id"),
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "related_images": chunk.metadata.get("related_images", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"切片保存到 {output_path}，共 {len(chunks)} 条")


def load_chunks_from_jsonl(input_path: str) -> List[Document]:
    """从 JSONL 加载切片为 Document 列表"""
    documents = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc = Document(
                page_content=record["content"],
                metadata={
                    "source": record.get("source", ""),
                    "chunk_id": record.get("chunk_id"),
                    "related_images": record.get("related_images", []),
                },
            )
            documents.append(doc)
    return documents
