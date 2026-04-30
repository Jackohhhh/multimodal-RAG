"""公共工具函数"""
import os
import re
from typing import List, Tuple
from langchain_core.documents import Document


def _friendly_source_label(source: str) -> str:
    """
    将内部 source 路径转换为对大模型友好的简短标签。
    例：'汇总英文手册.txt#Canon_Camera' → 'Canon_Camera'
        '相机手册.txt'                  → '相机手册'
    """
    basename = os.path.basename(source)
    if "#" in basename:
        # 多条目 JSONL：取 # 后面的产品名
        return basename.split("#", 1)[1]
    # 普通文件：去掉扩展名
    return os.path.splitext(basename)[0]


def format_docs(docs: List[Document]) -> str:
    """
    将检索到的文档格式化为上下文字符串。
    文本中的 [IMG:xxx] 标记保留，大模型可据此引用对应图片。
    不向大模型暴露检索分数，来源仅以友好标签呈现。
    """
    formatted = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "")
        label = _friendly_source_label(source) if source else f"片段{i+1}"
        content = doc.page_content

        formatted.append(
            f"[来源: {label}]:\n{content}"
        )

    return "\n\n".join(formatted)


def collect_image_ids(docs: List[Document]) -> List[str]:
    """从检索结果中收集所有关联的图片ID（去重保序）"""
    seen = set()
    result = []
    for doc in docs:
        for img_id in doc.metadata.get("related_images", []):
            if img_id not in seen:
                result.append(img_id)
                seen.add(img_id)
    return result


def postprocess_answer(answer: str) -> Tuple[str, List[str]]:
    """
    后处理大模型回答：
    1. 提取回答中所有 [IMG:xxx] 标记，按出现顺序收集图片ID
    2. 将 [IMG:xxx] 替换回 <PIC>
    返回: (含<PIC>的文本, 图片ID列表)
    """
    image_ids = re.findall(r'\[IMG:([^\]]+)\]', answer)
    text_with_pic = re.sub(r'\n?\[IMG:[^\]]+\]\n?', '<PIC>', answer)
    return text_with_pic, image_ids


def is_verbose_retrieval_enabled() -> bool:
    """读取检索调试开关，兼容常见拼写。"""
    value = (
        os.getenv("RAG_VERBOSE_RETRIEVAL")
        or os.getenv("RAG_VERBOSE_RETRIEVA")
        or ""
    ).strip().lower()
    return value in {"1", "true", "yes", "on"}


def preview_text(text: str, limit: int = 120) -> str:
    """生成单行预览，便于打印检索日志。"""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def log_retrieved_docs(
    query: str,
    docs: List[Document],
    *,
    pre_filter_hit_count: int | None = None,
    rag_relevance_threshold: float | None = None,
) -> None:
    """
    按环境变量控制输出检索命中摘要。
    默认在「阈值过滤之后」传入 docs，使 Hits 与进入 RAG 上下文的片段一致。
    """
    if not is_verbose_retrieval_enabled():
        return

    print("\n=== RAG Retrieval ===")
    print(f"Query: {preview_text(query, limit=200)}")
    if pre_filter_hit_count is not None and rag_relevance_threshold is not None:
        print(
            f"按 relevance_score > {rag_relevance_threshold} 过滤："
            f"原始 {pre_filter_hit_count} 条 → 进入 RAG 上下文 {len(docs)} 条"
        )
    print(f"Hits: {len(docs)}")
    if not docs:
        if pre_filter_hit_count and pre_filter_hit_count > 0:
            print(
                "（无片段超过阈值：不向 RAG 主模型传入手册片段；"
                "若仍生成回答，则来自客服兜底模型。）"
            )
        return
    for i, doc in enumerate(docs, start=1):
        score = doc.metadata.get("relevance_score", 0.0)
        source = doc.metadata.get("source", "未知")
        images = doc.metadata.get("related_images", [])
        print(
            f"[{i}] source={source} score={score:.2f} images={len(images)} "
            f"text={preview_text(doc.page_content)}"
        )
