"""
根据用户问题过滤检索片段的 source：
- 以英文为主的问题：仅保留「汇总英文手册」来源；
- 中文问题：须命中手册对应产品关键词，否则全部丢弃；保留的片段 source 须包含对应手册文件名子串。
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from langchain_core.documents import Document

# (问题中可能出现的词, metadata["source"] 中应包含的子串)
KEYWORD_SOURCE_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    sorted(
        [
            ("儿童电动摩托车", "儿童电动摩托车手册"),
            ("蓝牙激光鼠标", "蓝牙激光鼠标手册"),
            ("可编程温控器", "可编程温控器手册"),
            ("空气净化器", "空气净化器手册"),
            ("健身追踪器", "健身追踪器手册"),
            ("蒸汽清洁机", "蒸汽清洁机手册"),
            ("人体工学椅", "人体工学椅手册"),
            ("健身单车", "健身单车手册"),
            ("功能键盘", "功能键盘手册"),
            ("摩托艇", "摩托艇手册"),
            ("洗碗机", "洗碗机手册"),
            ("吹风机", "吹风机手册"),
            ("发电机", "发电机手册"),
            ("烤箱", "烤箱手册"),
            ("空调", "空调手册"),
            ("水泵", "水泵手册"),
            ("冰箱", "冰箱手册"),
            ("电钻", "电钻手册"),
            ("相机", "相机手册"),
            ("头显", "VR头显手册"),
            ("VR头显", "VR头显手册"),
            ("汇总英文", "汇总英文手册"),
        ],
        key=lambda x: len(x[0]),
        reverse=True,
    )
)

_ENGLISH_SUMMARY_SOURCE = "汇总英文手册"


def is_mainly_english_query(text: str) -> bool:
    """判断用户问题是否以英文为主（则只使用汇总英文手册）。"""
    s = (text or "").strip()
    if not s:
        return False
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    ascii_letters = sum(1 for c in s if c.isalpha() and ord(c) < 128)
    if ascii_letters >= 12 and cjk == 0:
        return True
    if ascii_letters > max(cjk, 1) * 4 and ascii_letters >= 10:
        return True
    return False


def required_source_substrings(question: str) -> Tuple[str, ...]:
    """
    返回允许的 source 子串。
    - 英文问题 -> (\"汇总英文手册\",)；
    - 中文命中产品词 -> 命中的若干子串（文档 source 包含其一即可）；
    - 未命中任何产品词 -> 空元组（调用方应丢弃全部片段）。
    """
    q = question or ""
    if is_mainly_english_query(q):
        return (_ENGLISH_SUMMARY_SOURCE,)

    normalized = re.sub(r"\s+", "", q)
    seen: List[str] = []
    for kw, src_sub in KEYWORD_SOURCE_PAIRS:
        if kw in normalized or kw in q:
            if src_sub not in seen:
                seen.append(src_sub)
    return tuple(seen)


def filter_documents_by_manual_source(
    question: str, documents: Sequence[Document]
) -> List[Document]:
    """无产品命中时返回空列表；否则只保留 source 命中允许子串的文档。"""
    required = required_source_substrings(question)
    if len(required) == 0:
        return []
    out: List[Document] = []
    for doc in documents:
        src = (doc.metadata.get("source") or "") if doc.metadata else ""
        if any(sub in src for sub in required):
            out.append(doc)
    return out
