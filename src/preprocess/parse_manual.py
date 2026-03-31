"""
解析说明书：处理 [text, [image_ids]] 格式的文档。
文本中 <PIC> 占位符按出现顺序与图片ID列表一一对应。
"""
import os
import re
import json
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document


def parse_pic_document(text: str, image_ids: List[str]) -> Tuple[str, Dict[int, str]]:
    """
    将文本中的 <PIC> 占位符替换为带图片ID的标记。
    第 N 个 <PIC> 对应 image_ids[N]。

    返回: (标记后的文本, {序号: 图片ID} 映射)
    """
    pic_count = text.count("<PIC>")
    if pic_count != len(image_ids):
        print(
            f"警告: 文本中有 {pic_count} 个 <PIC>，"
            f"但提供了 {len(image_ids)} 个图片ID，按较少的一方匹配"
        )

    mapping = {}
    marked_text = text
    for i in range(min(pic_count, len(image_ids))):
        img_id = image_ids[i]
        mapping[i] = img_id
        marked_text = marked_text.replace("<PIC>", f"\n[IMG:{img_id}]\n", 1)

    return marked_text, mapping


def _fix_json_escapes(raw: str) -> str:
    """修复 JSON 中不合法的转义序列，保留合法转义不变。"""
    _VALID = re.compile(
        r'(?P<valid>\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
        r'|'
        r'\\(?P<invalid>.)',
        re.DOTALL,
    )
    def _replacer(m):
        if m.group('valid'):
            return m.group('valid')
        return '\\\\' + m.group('invalid')
    return _VALID.sub(_replacer, raw)


def _parse_one(raw: str) -> Tuple[str, List[str]]:
    """解析单条 [text, [image_ids]] JSON 字符串。"""
    data = json.loads(raw)
    if isinstance(data, list) and len(data) == 2:
        text, image_ids = data[0], data[1]
        if isinstance(text, str) and isinstance(image_ids, list):
            return text, image_ids
    raise ValueError(
        f"文件格式不正确，期望 [text, [image_ids]]，实际得到: {type(data)}"
    )


def load_pic_document(file_path: str) -> List[Tuple[str, List[str]]]:
    """
    加载 [text, [image_ids]] 格式的文档文件。
    支持单条 JSON 和多行 JSONL（每行一条 JSON）。
    自动修复非法 JSON 转义序列。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = _fix_json_escapes(raw)

    try:
        return [_parse_one(raw)]
    except json.JSONDecodeError:
        pass

    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        results.append(_parse_one(line))
    if results:
        return results

    raise ValueError(f"无法解析文件: {file_path}")


def load_pic_documents_from_dir(dir_path: str) -> List[Dict]:
    """
    加载目录下所有 [text, [image_ids]] 格式的文档。
    支持 .json 和 .txt 后缀（内容均为 JSON 格式）。
    """
    results = []
    supported_exts = {".json", ".txt"}
    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_exts:
            continue
        try:
            docs = load_pic_document(file_path)
            for idx, (text, image_ids) in enumerate(docs):
                suffix = f"#{idx+1}" if len(docs) > 1 else ""
                doc_name = f"{filename}{suffix}"
                results.append({
                    "text": text,
                    "image_ids": image_ids,
                    "filename": doc_name,
                })
                print(f"已加载: {doc_name}，{text.count('<PIC>')} 张配图")
        except Exception as e:
            print(f"加载 {filename} 失败: {e}")
    return results


def build_documents_with_images(
    text: str,
    image_ids: List[str],
    source: str = "",
) -> Tuple[str, Dict[int, str]]:
    """
    解析单篇文档，返回 (标记后的完整文本, 图片映射)。
    标记后的文本中每个 <PIC> 已被替换为 [IMG:图片ID]。
    """
    marked_text, mapping = parse_pic_document(text, image_ids)
    return marked_text, mapping


def extract_image_ids_from_text(text: str) -> List[str]:
    """从标记文本中提取所有 [IMG:xxx] 中的图片ID"""
    return re.findall(r'\[IMG:([^\]]+)\]', text)
