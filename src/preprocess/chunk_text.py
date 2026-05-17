"""
文本切分：将标记后的文本切为「子块」建索引；可选按章节（Parent）保留整段上下文供 RAG。
先按说明书常见「章节标题」边界硬切，再在章内递归切分更小的 Child；可去掉行首 ``#``。
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

def _strip_line_heading_hash_prefix(line: str) -> str:
    """去掉行首 Markdown 标题的 ``#`` 前缀（不影响 ``[IMG:…]`` 行）。"""
    stripped = line.lstrip(" \t")
    if stripped.startswith("[IMG:"):
        return line
    if not stripped.startswith("#"):
        return line
    lead_ws_len = len(line) - len(stripped)
    prefix = line[:lead_ws_len]
    rest = stripped
    i = 0
    while i < len(rest) and rest[i] == "#":
        i += 1
    tail = rest[i:]
    if tail.startswith((" ", "\t")):
        tail = tail[1:]
    return prefix + tail


def strip_heading_hashes_multiline(text: str) -> str:
    """对全文逐行移除 ``#`` 章节标记（章节边界已由 ``_split_by_headers`` 处理完毕后再调用）。"""
    parts = [_strip_line_heading_hash_prefix(li) for li in text.splitlines()]
    return "\n".join(parts)


def extract_section_heading_hints(section: str, *, max_segments: int = 16) -> str:
    """在去掉 ``#`` 之前，抽取章节内标题行概要，写入 metadata 供标题补召回。"""
    hints: List[str] = []
    for line in (section or "").splitlines():
        s = line.strip()
        if not s or s.startswith("[IMG:"):
            continue
        if not s.startswith("#"):
            if hints:
                break
            continue
        for segment in s.split("#"):
            piece = segment.strip()
            if not piece:
                continue
            title = re.split(r"[。；;.!！?？\n]", piece, maxsplit=1)[0].strip()
            if title:
                hints.append(title[:80])
                if len(hints) >= max_segments:
                    return "\n".join(hints)
    return "\n".join(hints)


def _finalize_chunk_pipeline(chunks: List[str]) -> List[str]:
    """
    对同一批分段做：孤立图合并、段首图回移、「仅标题」块合并、[IMG:] 下移块尾。
    用于启用了 Parent-Chunk 时对「小节内」片段链做整形。
    """
    parts = _merge_orphan_img_sections(chunks)
    parts = _relocate_leading_img_lines_to_previous(parts)
    parts = _merge_heading_only_stacks_forward(parts)
    return [_move_leading_img_lines_to_chunk_end(s) for s in parts]


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
    strip_heading_hashes: bool = True,
    use_parent_document_retrieval: bool = False,
    child_chunk_size: Optional[int] = None,
    child_chunk_overlap: Optional[int] = None,
) -> tuple[List["Document"], List[dict]]:
    """
    切分含 [IMG:xxx] 标记的文本。

    策略：先按章节边界硬切（见 ``_split_by_headers``），合并孤立图块与
    ``# A.``/``# B.`` 等子节；再对超长段递归切分，并对子块合并孤儿图、图行回移与标题栈。
    可选 **Parent-Document Retrieval**：对每个章节块（Parent）用更小的子块（Child）建索引，
    Parent 正文由 ``save_chunks_to_jsonl`` 一并写入可选的 ``chunks_parents.jsonl``。
    ``strip_heading_hashes``：在分段与递归切完成后去掉行首 ``#``，减轻向量与模型噪声。
    可选丢弃含超长连续 ``.`` 的片段（目录引导线）。
    每个 chunk 的 metadata 含 ``related_images``；启用父文档时另有 ``parent_id``、``section_heading_hints``。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]

    sections = _split_by_headers(marked_text)

    eff_child_size = (
        int(child_chunk_size)
        if use_parent_document_retrieval and child_chunk_size is not None
        else chunk_size
    )
    eff_child_overlap = (
        int(child_chunk_overlap)
        if use_parent_document_retrieval and child_chunk_overlap is not None
        else chunk_overlap
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=eff_child_size,
        chunk_overlap=eff_child_overlap,
        separators=separators,
        keep_separator=True,
    )

    all_chunks: List[tuple[Optional[int], Optional[str], str]] = []

    parent_sidecar: List[dict] = []

    if use_parent_document_retrieval:
        for parent_id, section in enumerate(sections):
            hints_nl = extract_section_heading_hints(section) or ""
            hints_str = hints_nl if hints_nl else None

            if strip_heading_hashes:
                parent_plain = strip_heading_hashes_multiline(section.strip())
            else:
                parent_plain = section.strip()

            if (
                strip_dot_leader_lines
                and parent_plain
                and _chunk_has_excessive_dot_run(
                    parent_plain, max_allowed_consecutive=dot_run_max_allowed
                )
            ):
                continue

            p_imgs = re.findall(r"\[IMG:([^\]]+)\]", parent_plain)
            parent_sidecar.append(
                {
                    "parent_id": parent_id,
                    "content": parent_plain,
                    "related_images": p_imgs,
                    "section_heading_hints": hints_nl,
                }
            )

            raw_list = (
                [section]
                if len(section) <= eff_child_size
                else text_splitter.split_text(section)
            )
            finalized = _finalize_chunk_pipeline(raw_list)

            for child_text in finalized:
                core = (child_text or "").strip()
                if not core:
                    continue
                all_chunks.append((parent_id, hints_str, core))
    else:
        for section in sections:
            if len(section) <= chunk_size:
                all_chunks.append((None, None, section))
            else:
                for sub in text_splitter.split_text(section):
                    all_chunks.append((None, None, sub))

        str_chunks = [item[2] for item in all_chunks]
        str_chunks = _merge_orphan_img_sections(str_chunks)
        str_chunks = _relocate_leading_img_lines_to_previous(str_chunks)
        str_chunks = _merge_heading_only_stacks_forward(str_chunks)
        str_chunks = [
            _move_leading_img_lines_to_chunk_end(s) for s in str_chunks
        ]
        all_chunks = [(None, None, s) for s in str_chunks]

    merged_out = [txt for _, _, txt in all_chunks]

    meta_parent_ids = [pid for pid, _, _ in all_chunks]
    meta_parent_hints = [h for _, h, _ in all_chunks]

    if strip_heading_hashes:
        merged_out = [strip_heading_hashes_multiline(x) for x in merged_out]

    if strip_dot_leader_lines:
        kept: List[tuple[str, Optional[int], Optional[str]]] = []
        dropped_dot = 0
        for chunk_text, pid, hint in zip(merged_out, meta_parent_ids, meta_parent_hints):
            if _chunk_has_excessive_dot_run(
                chunk_text, max_allowed_consecutive=dot_run_max_allowed
            ):
                dropped_dot += 1
                continue
            kept.append((chunk_text, pid, hint))
        if dropped_dot:
            print(
                f"  [{source}] 含连续句点超过 {dot_run_max_allowed} 个的片段整段丢弃: {dropped_dot} 个"
            )
        merged_out = [k[0] for k in kept]
        meta_parent_ids = [k[1] for k in kept]
        meta_parent_hints = [k[2] for k in kept]

    # 过滤极短（< 8 字符）且无图片标记的孤立 chunk
    kept_nonempty: List[str] = []
    kept_pid: List[Optional[int]] = []
    kept_hints: List[Optional[str]] = []
    dropped_short = 0
    for i, chunk_text in enumerate(merged_out):
        stripped = chunk_text.strip()
        if len(stripped) < 8 and not re.search(r"\[IMG:", stripped):
            dropped_short += 1
            continue
        kept_nonempty.append(chunk_text)
        kept_pid.append(meta_parent_ids[i])
        kept_hints.append(meta_parent_hints[i])
    if dropped_short:
        print(f"  [{source}] 丢弃极短无效片段: {dropped_short} 个")

    documents: List[Document] = []
    for i, chunk_text in enumerate(kept_nonempty):
        image_ids = re.findall(r"\[IMG:([^\]]+)\]", chunk_text)
        meta: dict = {
            "source": source,
            "chunk_id": i,
            "related_images": image_ids,
        }
        pid = kept_pid[i]
        if pid is not None:
            meta["parent_id"] = pid
            h = kept_hints[i]
            if h:
                meta["section_heading_hints"] = h
        documents.append(
            Document(
                page_content=chunk_text,
                metadata=meta,
            )
        )

    total_images = sum(len(d.metadata["related_images"]) for d in documents)
    mode = (
        f"父子索引(child={eff_child_size}±{eff_child_overlap})"
        if use_parent_document_retrieval
        else f"单粒度(size={chunk_size})"
    )
    print(
        f"切分完成 ({mode}): {len(documents)} 个片段，"
        f"其中 {sum(1 for d in documents if d.metadata['related_images'])} 个含图片，"
        f"共关联 {total_images} 张图"
    )
    return documents, parent_sidecar


def save_chunks_to_jsonl(
    chunks: List["Document"],
    output_path: str,
    *,
    parents: Optional[List[dict]] = None,
    parents_output_path: Optional[str] = None,
):
    """将子片段写入 JSONL；可选同时写入 Parent 上下文（chunks_parents.jsonl）。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            meta = chunk.metadata or {}
            record: dict = {
                "chunk_id": meta.get("chunk_id"),
                "content": chunk.page_content,
                "source": meta.get("source", ""),
                "related_images": meta.get("related_images", []),
            }
            pid = meta.get("parent_id")
            if pid is not None:
                record["parent_id"] = pid
            sh = meta.get("section_heading_hints")
            if sh:
                record["section_heading_hints"] = sh
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"切片保存到 {output_path}，共 {len(chunks)} 条")

    rows = parents or []
    if rows and parents_output_path:
        os.makedirs(os.path.dirname(parents_output_path) or ".", exist_ok=True)
        with open(parents_output_path, "w", encoding="utf-8") as pf:
            for row in rows:
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Parent 上下文保存到 {parents_output_path}，共 {len(rows)} 条")


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
            meta: dict = {
                "source": record.get("source", ""),
                "chunk_id": record.get("chunk_id"),
                "related_images": record.get("related_images", []),
            }
            pid = record.get("parent_id")
            if pid is not None:
                meta["parent_id"] = pid
            sh = record.get("section_heading_hints")
            if sh:
                meta["section_heading_hints"] = sh
            doc = Document(
                page_content=record["content"],
                metadata=meta,
            )
            documents.append(doc)
    return documents


def load_parents_from_jsonl(input_path: str) -> dict[tuple[str, int], dict]:
    """
    {(source basename or key string, parent_id): {"content", "related_images", "section_heading_hints"}}
    """
    lookup: dict[tuple[str, int], dict] = {}
    if not input_path or not os.path.isfile(input_path):
        return lookup
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            src = record.get("source") or ""
            pid = record.get("parent_id")
            if pid is None:
                continue
            lookup[(src, int(pid))] = {
                "content": record.get("content") or "",
                "related_images": record.get("related_images") or [],
                "section_heading_hints": record.get("section_heading_hints") or "",
            }
    return lookup
