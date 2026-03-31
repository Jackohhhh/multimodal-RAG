"""
多模态输入处理：解析用户上传的图片，用多模态模型转为文本描述后辅助检索。
"""
import base64
import os
from typing import Optional

from langchain_core.messages import HumanMessage


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """根据文件扩展名返回 MIME 类型"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def describe_image(image_path: str, multimodal_llm) -> str:
    """
    用多模态大模型为用户上传的图片生成文字描述，
    以便转化为文本 query 进行知识库检索。
    """
    b64_image = encode_image_to_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "请仔细观察这张图片，描述你看到的产品、故障现象、"
                    "操作界面或指示灯状态等关键信息。"
                    "用简洁的中文输出，以便后续检索产品说明书。"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_image}",
                },
            },
        ]
    )
    response = multimodal_llm.invoke([message])
    return response.content


def build_multimodal_query(
    text_query: str,
    image_path: Optional[str],
    multimodal_llm,
) -> str:
    """
    构建多模态查询：如果用户同时提供了文字和图片，
    将图片描述拼接到文字查询中，增强检索精度。
    """
    if not image_path:
        return text_query

    image_desc = describe_image(image_path, multimodal_llm)
    return f"{text_query}\n\n【用户上传图片描述】：{image_desc}"
