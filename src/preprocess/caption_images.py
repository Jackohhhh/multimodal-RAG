"""
图片 Caption 生成：用多模态大模型为说明书配图生成文字描述。
生成的 caption 可以被索引到向量库中，支持"以文搜图"。
"""
import os
import json
import base64
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage


DEFAULT_CAPTION_PROMPT = (
    "你是一个产品说明书图片分析专家。请仔细观察图片，"
    "用准确、简洁的中文描述图片内容，包括：\n"
    "1. 图片展示的是什么（产品部件、操作步骤、界面截图、故障现象等）\n"
    "2. 图中的关键标注、箭头指向、数字编号\n"
    "3. 如果有文字，请完整转录\n"
    "输出纯文本描述，不超过 300 字。"
)


def generate_caption(
    image_path: str,
    multimodal_llm,
    prompt: Optional[str] = None,
) -> str:
    """为单张图片生成文字描述。prompt 未传入时使用默认值。"""
    caption_prompt = prompt or DEFAULT_CAPTION_PROMPT

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    message = HumanMessage(
        content=[
            {"type": "text", "text": caption_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
        ]
    )
    response = multimodal_llm.invoke([message])
    return response.content


def batch_generate_captions(
    image_dir: str,
    output_dir: str,
    multimodal_llm,
    prompt: Optional[str] = None,
    supported_exts: Optional[List[str]] = None,
) -> List[Dict]:
    """
    批量为目录下的图片生成 caption。
    返回 caption 记录列表，并保存为 JSON。
    """
    if supported_exts is None:
        supported_exts = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

    os.makedirs(output_dir, exist_ok=True)
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

    captions = []
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        print(f"[{i+1}/{len(image_files)}] 生成 caption: {filename}")
        try:
            caption = generate_caption(image_path, multimodal_llm, prompt=prompt)
            record = {
                "image_filename": filename,
                "image_path": image_path,
                "caption": caption,
            }
            captions.append(record)
        except Exception as e:
            print(f"  生成失败: {e}")
            captions.append({
                "image_filename": filename,
                "image_path": image_path,
                "caption": "",
                "error": str(e),
            })

    output_path = os.path.join(output_dir, "captions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"Caption 生成完成，结果保存到 {output_path}")

    return captions
