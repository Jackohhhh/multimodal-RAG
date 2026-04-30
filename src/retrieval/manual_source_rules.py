"""
根据用户问题过滤检索片段的 source（专一性）：
- 中英文问题：若命中产品关键词，优先仅保留对应手册条目的片段；
- 英文问题：若未命中具体产品关键词，则退回「汇总英文手册」来源；
- 中文问题：若未命中任何产品关键词，不做 source 限制，保留检索器返回的全部片段。
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
            # 英文手册各条目：产品关键词/常见问法 → source 后缀子串
            ("over-the-range microwave", "Microwave_Oven"),
            ("multi-use pressure cooker and air fryer", "Pressure_Cooker_Air_Fryer"),
            ("pressure cooker and air fryer", "Pressure_Cooker_Air_Fryer"),
            ("dual-mode virtual wall barrier", "Vacuum_Cleaner"),
            ("dual mode virtual wall barrier", "Vacuum_Cleaner"),
            ("virtual wall barrier", "Vacuum_Cleaner"),
            ("front caster wheel", "Vacuum_Cleaner"),
            ("full bin sensors", "Vacuum_Cleaner"),
            ("charging contacts", "Vacuum_Cleaner"),
            ("robot anatomy", "Vacuum_Cleaner"),
            ("home base", "Vacuum_Cleaner"),
            ("extractors", "Vacuum_Cleaner"),
            ("side brush", "Vacuum_Cleaner"),
            ("T-rail", "Network_Camera"),
            ("t-rail", "Network_Camera"),
            ("power the camera", "Network_Camera"),
            ("camera battery", "Canon_Camera"),
            ("date/time battery", "Canon_Camera"),
            ("household power socket", "Canon_Camera"),
            ("off-center subject", "Canon_Camera"),
            ("eyepiece cover", "Canon_Camera"),
            ("AF Mode", "Canon_Camera"),
            ("CP direct", "Canon_Camera"),
            ("CF card", "Canon_Camera"),
            ("Mode Dial", "Canon_Camera"),
            ("lens", "Canon_Camera"),
            ("snowmobile", "Motorcycle"),
            ("V-Belt", "Motorcycle"),
            ("v-belt", "Motorcycle"),
            ("brake lever", "Motorcycle"),
            ("brake button", "Motorcycle"),
            ("throttle cable", "Motorcycle"),
            ("spark plug", "Motorcycle"),
            ("jetski", "WaveRunner_2005"),
            ("jet ski", "WaveRunner_2005"),
            ("watercraft", "WaveRunner_2005"),
            ("Quick Shift Trim System", "WaveRunner_2005"),
            ("QSTS", "WaveRunner_2005"),
            ("fuel meter", "WaveRunner_2005"),
            ("hour meter", "WaveRunner_2005"),
            ("filler cap", "WaveRunner_2005"),
            ("battery conversion", "Boat_210FSH"),
            ("battery switch", "Boat_210FSH"),
            ("storage compartments", "Boat_210FSH"),
            ("battery compartment", "Boat_210FSH"),
            ("anchor light", "Boat_210FSH"),
            ("jet wash", "Boat_210FSH"),
            ("water supply", "Boat_210FSH"),
            ("bimini top", "Boat_210FSH"),
            ("bilge pump", "Boat_210FSH"),
            ("livewell", "Boat_210FSH"),
            ("swim platform", "Boat_210FSH"),
            ("boat controller", "Boat_210FSH"),
            ("trip screen", "Boat_210FSH"),
            ("fire extinguisher", "Boat_210FSH"),
            ("throttle-cable", "Boat_210FSH"),
            ("steering system", "Boat_210FSH"),
            ("cooling system", "Boat_210FSH"),
            ("engine oil level", "Boat_210FSH"),
            ("boat", "Boat_210FSH"),
            ("ship", "Boat_210FSH"),
            ("sailing", "Boat_210FSH"),
            ("coffee machine", "Espresso_Machine"),
            ("coffee maker", "Espresso_Machine"),
            ("factory settings", "Espresso_Machine"),
            ("energy saving", "Espresso_Machine"),
            ("water volume", "Espresso_Machine"),
            ("empty the system", "Espresso_Machine"),
            ("airfryer", "Air_Fryer_Philips"),
            ("air fryer", "Air_Fryer_Philips"),
            ("earphones", "ANC_Earphones"),
            ("earbuds", "ANC_Earphones"),
            ("sound system", "ANC_Earphones"),
            ("bluetooth", "ANC_Earphones"),
            ("eReader", "Ebook_Device"),
            ("e-book reader", "Ebook_Device"),
            ("Browser History", "Ebook_Device"),
            ("photo viewer", "Ebook_Device"),
            ("fax", "Power_Tool_Safety"),
            ("LP Tank", "Outdoor_Grill"),
            ("regulator", "Outdoor_Grill"),
            ("indirect cooking", "Outdoor_Grill"),
            ("assembly process", "Outdoor_Grill"),
            ("landline", "Answering_Machine"),
            ("base station", "Answering_Machine"),
            ("handset", "Answering_Machine"),
            ("LED indicator", "Answering_Machine"),
            ("lawn mower", "Outdoor_Engine_Generator"),
            ("roll bar", "Outdoor_Engine_Generator"),
            ("height of cut", "Outdoor_Engine_Generator"),
            ("mower belt", "Outdoor_Engine_Generator"),
            ("rear-shock", "Outdoor_Engine_Generator"),
            ("motherboard", "Computer_Manual"),
            ("PCI Express", "Computer_Manual"),
            ("BIOS", "Computer_Manual"),
            ("RAID", "Computer_Manual"),
            ("CPU", "Computer_Manual"),
            ("TPM", "Computer_Manual"),
            ("television", "Color_Television"),
            ("TV", "Color_Television"),
            ("DVD player", "Color_Television"),
            ("Closed Captions", "Color_Television"),
            ("captions", "Color_Television"),
            ("outdoor antenna", "Color_Television"),
            ("toothbrush", "Electric_Toothbrush"),
            ("Canon Camera", "Canon_Camera"),
            ("Espresso", "Espresso_Machine"),
            ("espresso", "Espresso_Machine"),
            ("Air Fryer", "Air_Fryer_Philips"),
            ("Airfry", "Air_Fryer_Philips"),
            ("210FSH", "Boat_210FSH"),
            ("WaveRunner", "WaveRunner_2005"),
            ("chainsaw", "Power_Tool_Safety"),
            ("earphone", "ANC_Earphones"),
            ("earbud", "ANC_Earphones"),
            ("ebook", "Ebook_Device"),
            ("Grill", "Outdoor_Grill"),
            ("grill", "Outdoor_Grill"),
            ("Motorcycle", "Motorcycle"),
            ("motorcycle", "Motorcycle"),
            ("Television", "Color_Television"),
            ("Vacuum", "Vacuum_Cleaner"),
            ("vacuum", "Vacuum_Cleaner"),
            ("Network Camera", "Network_Camera"),
            ("Toothbrush", "Electric_Toothbrush"),
            ("toothbrush", "Electric_Toothbrush"),
            ("Washing Machine", "Washing_Machine"),
            ("Pressure Cooker", "Pressure_Cooker_Air_Fryer"),
            ("Microwave", "Microwave_Oven"),
            ("microwave", "Microwave_Oven"),
            ("Answering Machine", "Answering_Machine"),
            ("Generator", "Outdoor_Engine_Generator"),
            ("generator", "Outdoor_Engine_Generator"),
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
    返回允许的 source 子串；空元组表示「不按 source 过滤」。
    - 中英文命中产品词 -> 命中的若干子串（文档 source 包含其一即可）；
    - 英文未命中产品词 -> (\"汇总英文手册\",)；
    - 中文未命中产品词 -> 空元组。
    """
    q = question or ""
    normalized = re.sub(r"\s+", "", q)
    q_lower = q.lower()
    normalized_lower = normalized.lower()
    seen: List[str] = []
    for kw, src_sub in KEYWORD_SOURCE_PAIRS:
        kw_lower = kw.lower()
        if kw in normalized or kw in q or kw_lower in q_lower or kw_lower in normalized_lower:
            if src_sub not in seen:
                seen.append(src_sub)
    if seen:
        return tuple(seen)
    if is_mainly_english_query(q):
        return (_ENGLISH_SUMMARY_SOURCE,)
    return tuple(seen)


def filter_documents_by_manual_source(
    question: str, documents: Sequence[Document]
) -> List[Document]:
    """有明确 source 约束时只保留匹配文档；无约束时原样返回全部检索结果。"""
    required = required_source_substrings(question)
    if len(required) == 0:
        return list(documents)
    out: List[Document] = []
    for doc in documents:
        src = (doc.metadata.get("source") or "") if doc.metadata else ""
        if any(sub in src for sub in required):
            out.append(doc)
    return out
