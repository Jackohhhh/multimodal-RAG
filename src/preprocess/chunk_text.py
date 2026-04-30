"""
文本切分：将标记后的文本切分为片段，每个片段自动携带其包含的图片ID。
先按说明书常见「章节标题」边界硬切，再对超长段做递归切分。
"""
import os
import re
import json
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

# # 或 ## 后接：空格类 / 全角空格；或直接接标题首字（中文、字母、数字、括号）
_TITLE_AFTER_HASH = r"#+(?:[\s\u3000]+|[\u4e00-\u9fffA-Za-z0-9（\(])"

# 英文手册全大写标题模式：2+ 个全大写单词（可含 /、- 等）
# 例：FAVORITE RECIPE / HOLD WARM / EASY COOK / CHILD LOCK
# 排除：单个全大写词（NOTE / WARNING / CAUTION 等独立词不作分割）
_ALLCAPS_SECTION_TITLE = (
    r"(?<!\w)"                                              # 前面不是普通字符
    r"[A-Z][A-Z0-9/\-]+"                                   # 第一个全大写单词（2+字符）
    r"(?:\s+[A-Z][A-Z0-9/\-]+)+"                           # 1+ 个后续全大写单词
)

# 零宽切分：在「新章节」起点前断开（不把正文吞进标题里）
_SECTION_BOUNDARY = re.compile(
    # 行首 / 全文开头的 Markdown 标题（含 #标题 无空格）
    rf"(?=(?:(?:\A)|(?:\n)){_TITLE_AFTER_HASH})"
    # 图片占位展开后常见形态：换行 [IMG:…] 换行 再跟 # 章节
    rf"|(?=\n\[IMG:[^\]]+\]\n\s*{_TITLE_AFTER_HASH})"
    # 句、段结束标点后的同段内 # 标题（空调类手册常见）
    rf"|(?<=[。！？…．;；])(?=\s*{_TITLE_AFTER_HASH})"
    rf"|(?<=[.!?])(?=\s*{_TITLE_AFTER_HASH})"
    # 行内「…字或标点 # 标题」（吹风机等 <PIC># 连成串、预处理后才断行的情况）
    rf"|(?<=[\u4e00-\u9fff0-9\)）>」』\s])(?=\s*{_TITLE_AFTER_HASH})"
    # 英文手册全大写节标题（段末标点+双空格后 / 换行后）
    # 匹配：...cooking.  FAVORITE RECIPE  / 换行开头的 FAVORITE RECIPE
    rf"|(?<=\.\s\s)(?={_ALLCAPS_SECTION_TITLE})"
    rf"|(?=\n(?:{_ALLCAPS_SECTION_TITLE})\s)"
)

# 仅含图片占位、无说明正文的段（预处理常见），并入相邻段以免检索碎片
_ORPHAN_IMG_BLOCK = re.compile(r"^(\s*\[IMG:[^\]]+\]\s*)+$")

def _chunk_has_excessive_dot_run(text: str, max_allowed_consecutive: int = 10) -> bool:
    """
    是否存在「超过 max_allowed_consecutive 个」连续引导点。
    同时检测：
    - ASCII 句号 ``.``（U+002E）
    - Unicode 省略号 ``…``（U+2026，常见于英文手册 OCR 提取的目录引导线）
    默认 10 即连续 11 个及以上时返回 True（整块应丢弃）。
    """
    n = int(max_allowed_consecutive)
    if n < 1:
        return False
    # ASCII 句号连续
    if re.search(rf"\.{{{n + 1},}}", text):
        return True
    # Unicode 省略号连续（每个 … 视觉上等于 3 个点，3 个 … 即 9 个点，阈值取 3）
    ellipsis_threshold = max(3, n // 3)
    if re.search(rf"\u2026{{{ellipsis_threshold},}}", text):
        return True
    return False


def _merge_orphan_img_sections(sections: List[str]) -> List[str]:
    """将「只有 [IMG:…] 行」的短段合并进下一段；若段末仍有悬空图块则并回上一段。"""
    merged: List[str] = []
    pending = ""
    for raw in sections:
        s = raw.strip()
        if not s:
            continue
        if _ORPHAN_IMG_BLOCK.fullmatch(s):
            pending = f"{pending}\n{s}".strip() if pending else s
            continue
        if pending:
            merged.append(f"{pending}\n{s}".strip())
            pending = ""
        else:
            merged.append(s)
    if pending:
        if merged:
            merged[-1] = f"{merged[-1].rstrip()}\n{pending}".strip()
        else:
            merged.append(pending)
    return merged


def _first_hash_heading_line(text: str) -> str:
    """跳过段首纯 [IMG:…] 行，返回第一个以 # 开头的标题行。"""
    for line in text.strip().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[IMG:"):
            continue
        if s.startswith("#"):
            return s
    return ""


def _is_numbered_main_heading_block(text: str) -> bool:
    """首条 # 标题是否为「数字编号主节」（# 1. # 7. # 9.2 等）。"""
    fh = _first_hash_heading_line(text)
    return bool(fh and re.match(r"^#+\s*\d", fh))


def _is_subordinate_heading_block(text: str) -> bool:
    """
    是否为应并入上一「编号主节」的子节标题块：
    - 单行内连写的 # 7. … # A. … 被边界规则拆开后，# A.、# B. 等应跟在同一 chunk；
    - 保修等章节下的 # I. # II. 等同理。
    非子节：仍以 # 数字 开头的块。
    """
    fh = _first_hash_heading_line(text)
    if not fh:
        return False
    if re.match(r"^#+\s*\d", fh):
        return False
    # 单子母 + 点（全角点等）：# A. # B. …
    if re.match(r"^#+\s*[A-Za-zＡ-Ｚ]\s*[\.\s．、]", fh):
        return True
    # I. II. III.
    if re.match(r"^#+\s*I+\.", fh):
        return True
    # 常见罗马数字节标题
    if re.match(
        r"^#+\s*(?:IV|VI{1,3}|IX|XI{1,3}|XIV|XIX|XX|XXX)\s*[\.\s．、]",
        fh,
    ):
        return True
    return False


def _block_can_host_subordinate_merge(text: str) -> bool:
    """上一段是否允许再接 # A. / # I. 类子节（编号主节或已是子节链）。"""
    return _is_numbered_main_heading_block(text) or _is_subordinate_heading_block(
        text
    )


def _merge_subordinate_hash_sections(sections: List[str]) -> List[str]:
    """
    将 # A. # B. # I. 等子节与紧前一段合并（同一大节内被「行内 #」规则误切时）。
    """
    merged: List[str] = []
    for raw in sections:
        s = raw.strip()
        if not s:
            continue
        if (
            merged
            and _is_subordinate_heading_block(s)
            and _block_can_host_subordinate_merge(merged[-1])
        ):
            merged[-1] = f"{merged[-1].rstrip()}\n{s}".strip()
        else:
            merged.append(s)
    return merged


def _should_relocate_leading_imgs_after_line(first_meaningful_line: str) -> bool:
    """
    段首若干行 [IMG:] 之后，若正文属于「新小节」而非紧接上一句，应把图挪到上一段末尾。
    - Markdown 标题 # …
    - 英文说明书编号步骤：9 When… / 10 Pull…（含 OCR 成 9When 的情况）
    """
    s = first_meaningful_line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    # 数字 + 可选空格 + 字母起句（步骤/小节）
    if re.match(r"^\d+\s*[A-Za-z]", s):
        return True
    return False


def _move_leading_img_lines_to_chunk_end(text: str) -> str:
    """
    将块首连续若干行 [IMG:…] 移到该块末尾（正文优先），便于向量检索以首句语义为主，
    同时块内仍保留全部 [IMG:] 供 RAG 提示词引用。metadata.related_images 由全文正则提取，与顺序无关。
    """
    lines = text.splitlines()
    lead_imgs: List[str] = []
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if not t:
            i += 1
            continue
        if t.startswith("[IMG:"):
            lead_imgs.append(lines[i])
            i += 1
        else:
            break
    if not lead_imgs:
        return text
    rest_lines = lines[i:]
    if not any(x.strip() for x in rest_lines):
        return text
    rest = "\n".join(rest_lines).rstrip()
    tail = "\n".join(lead_imgs).strip()
    # 只 rstrip 全文，避免 strip 吃掉正文第一行的前导空格/缩进
    return f"{rest}\n\n{tail}".rstrip()


def _relocate_leading_img_lines_to_previous(chunks: List[str]) -> List[str]:
    """
    若某段以连续 [IMG:…] 行开头，且首条正文像「新起一节」（# 标题或英文编号步骤），
    则将图行移到上一段末尾，避免图挂在下一段开头。
    """
    if len(chunks) <= 1:
        return chunks
    out: List[str] = [chunks[0]]
    for s in chunks[1:]:
        lines = s.splitlines()
        lead_imgs: List[str] = []
        i = 0
        while i < len(lines):
            t = lines[i].strip()
            if not t:
                i += 1
                continue
            if t.startswith("[IMG:"):
                lead_imgs.append(lines[i])
                i += 1
            else:
                break
        if not lead_imgs:
            out.append(s)
            continue
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            out.append(s)
            continue
        first_text = lines[i].strip()
        if not _should_relocate_leading_imgs_after_line(first_text):
            out.append(s)
            continue
        rest = "\n".join(lines[i:]).strip()
        out[-1] = (out[-1].rstrip() + "\n" + "\n".join(lead_imgs)).strip()
        out.append(rest)
    return out


def _chunk_is_heading_stack_only(text: str, max_chars: int = 400) -> bool:
    """
    块内是否仅有 Markdown 标题行（可多条 # 行）与空行，无正文、无 [IMG:]。
    用于识别「孤立标题」碎片（如 # 蒸汽清洁机组装说明 单独成块、# Belt Maintenance）。
    """
    t = text.strip()
    if not t or len(t) > max_chars:
        return False
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[IMG:"):
            return False
        if not s.startswith("#"):
            return False
    return True


def _merge_heading_only_stacks_forward(chunks: List[str]) -> List[str]:
    """
    将「仅含 # 标题行」的短块并入下一段；若最后一块如此则并回上一段。
    解决行内 # 边界把「主标题」与「# 无需专用工具」等拆开、或英文小节标题单独成块的问题。
    """
    if len(chunks) < 2:
        return chunks

    out = list(chunks)
    changed = True
    rounds = 0
    while changed and rounds < 100:
        changed = False
        rounds += 1
        new_out: List[str] = []
        i = 0
        while i < len(out):
            s = out[i]
            if i + 1 < len(out) and _chunk_is_heading_stack_only(s):
                new_out.append(f"{s.strip()}\n{out[i + 1].strip()}".strip())
                i += 2
                changed = True
            else:
                new_out.append(s)
                i += 1
        out = new_out
        if len(out) >= 2 and _chunk_is_heading_stack_only(out[-1]):
            out[-2] = f"{out[-2].rstrip()}\n{out[-1].strip()}".strip()
            out.pop()
            changed = True
    return out


def _split_by_headers(text: str) -> List[str]:
    """
    按章节边界切分：优先对齐「# / ## 标题」在手册中的各种出现形式。

    覆盖：
    - 行首 ``# 标题``、``## 标题``、``#标题``（无空格）；
    - ``\\n[IMG:…]\\n`` 后的标题（图文交错排版）；
    - 句号、叹号、分号等后的 ``# 标题``（同一段内多级标题）；
    - 中文叙述后空格再接 ``# 标题``（原稿 ``<PIC>#`` 等挤在一行的情况）。
    """
    parts = _SECTION_BOUNDARY.split(text)
    sections: List[str] = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            sections.append(stripped)
    sections = _merge_orphan_img_sections(sections)
    sections = _merge_subordinate_hash_sections(sections)
    sections = _relocate_leading_img_lines_to_previous(sections)
    return _merge_heading_only_stacks_forward(sections)


def chunk_marked_text(
    marked_text: str,
    source: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
    *,
    strip_dot_leader_lines: bool = True,
    dot_run_max_allowed: int = 10,
) -> List["Document"]:
    """
    切分含 [IMG:xxx] 标记的文本。

    策略：先按章节边界硬切（见 ``_split_by_headers``），合并孤立图块与
    ``# A.``/``# B.`` 等子节，并将「段首图 + 下接 # 标题或英文编号步骤」的图行移回上一段末尾，
    再合并「仅含 # 标题行」的孤立短块；仅对超长段递归切分，并对子块再次合并孤儿图、图行回移与标题合并。
    最后将「段首连续 [IMG:]、后接正文」的图行挪到块末，减轻英文手册中图标记挤在检索文本开头的问题。
    可选：若块内出现「超过 dot_run_max_allowed 个」连续 ``.``（目录引导线等），整段丢弃。
    每个 chunk 的 metadata 自动包含该 chunk 内出现的图片ID列表。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

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

    # 递归切分可能产生「仅 [IMG:]」短块，需与章节级同样先合并再回移图行
    all_chunks = _merge_orphan_img_sections(all_chunks)
    all_chunks = _relocate_leading_img_lines_to_previous(all_chunks)
    all_chunks = _merge_heading_only_stacks_forward(all_chunks)
    all_chunks = [_move_leading_img_lines_to_chunk_end(s) for s in all_chunks]

    if strip_dot_leader_lines:
        kept: List[str] = []
        dropped_dot = 0
        for s in all_chunks:
            if _chunk_has_excessive_dot_run(s, max_allowed_consecutive=dot_run_max_allowed):
                dropped_dot += 1
                continue
            kept.append(s)
        if dropped_dot:
            print(
                f"  [{source}] 含连续句点超过 {dot_run_max_allowed} 个的片段整段丢弃: {dropped_dot} 个"
            )
        all_chunks = kept

    # 过滤极短（< 8 字符）且无图片标记的孤立 chunk（如单独的句号、纯标题行、空白段）
    kept_nonempty: List[str] = []
    dropped_short = 0
    for s in all_chunks:
        stripped = s.strip()
        if len(stripped) < 8 and not re.search(r'\[IMG:', stripped):
            dropped_short += 1
            continue
        kept_nonempty.append(s)
    if dropped_short:
        print(f"  [{source}] 丢弃极短无效片段: {dropped_short} 个")
    all_chunks = kept_nonempty

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


def save_chunks_to_jsonl(chunks: List["Document"], output_path: str):
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


def load_chunks_from_jsonl(input_path: str) -> List["Document"]:
    """从 JSONL 加载切片为 Document 列表"""
    from langchain_core.documents import Document

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
