"""
项目统一入口脚本。
支持多种运行模式：预处理、交互对话、批量评测。
"""
import argparse
import logging
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1") #加了 HF_HUB_OFFLINE=1 后，启动时直接读本地缓存，不再联网检查 HuggingFace。启动速度也会快一些
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def cmd_preprocess(args):
    """数据预处理：解析说明书（<PIC>占位符格式）、切分、构建向量索引"""
    from src.preprocess.parse_manual import (
        load_pic_documents_from_dir, build_documents_with_images,
    )
    from src.preprocess.chunk_text import chunk_marked_text, save_chunks_to_jsonl
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from pymilvus import connections, MilvusClient
    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = os.path.expandvars(raw)
    config = yaml.safe_load(raw)

    data_cfg = config["data"]

    # --- 步骤 1: 加载文档（[text, [image_ids]] 格式）---
    print("=== 步骤 1: 加载说明书 ===")
    raw_docs = load_pic_documents_from_dir(data_cfg["manuals_path"])
    if not raw_docs:
        print("未找到文档，请检查 manuals_path 目录。")
        return

    # --- 步骤 2: 解析 <PIC> 占位符，替换为 [IMG:图片ID] ---
    print("\n=== 步骤 2: 解析图文映射 ===")
    all_chunks = []
    for doc_info in raw_docs:
        marked_text, mapping = build_documents_with_images(
            text=doc_info["text"],
            image_ids=doc_info["image_ids"],
            source=doc_info["filename"],
        )
        pic_count = len(mapping)
        print(f"  {doc_info['filename']}: {pic_count} 个 <PIC> 已映射为 [IMG:id]")

        # --- 步骤 3: 切分（保留 [IMG:id] 标记）---
        chunks = chunk_marked_text(
            marked_text=marked_text,
            source=doc_info["filename"],
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )
        all_chunks.extend(chunks)

    print(f"\n总计: {len(all_chunks)} 个片段")
    save_chunks_to_jsonl(all_chunks, data_cfg["chunks_output"])

    # --- 步骤 4: 构建向量索引 ---
    print("\n=== 步骤 3: 构建向量索引 ===")
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embeddings"]["model_name"]
    )

    conn_args = config["vector_store"]["connection_args"]
    collection_name = config["vector_store"]["collection_name"]

    from pymilvus import utility

    bootstrap_client = MilvusClient(**conn_args)
    alias = bootstrap_client._using
    if not connections.has_connection(alias):
        connections.connect(alias=alias, **conn_args)

    if utility.has_collection(collection_name, using=alias):
        utility.drop_collection(collection_name, using=alias)
        print(f"已删除旧集合 '{collection_name}'")

    vector_store = Milvus(
        connection_args=conn_args,
        collection_name=collection_name,
        embedding_function=embeddings,
        auto_id=True,
        drop_old=False,
    )

    import json as _json
    for doc in all_chunks:
        imgs = doc.metadata.get("related_images")
        if isinstance(imgs, list):
            doc.metadata["related_images"] = _json.dumps(imgs, ensure_ascii=False)

    vector_store.add_documents(all_chunks)
    print(f"已将 {len(all_chunks)} 个片段索引到 Milvus 集合 '{collection_name}'")

    print("\n预处理完成！")


def cmd_chat(args):
    """交互式对话模式"""
    import json
    from src.agent.agent import CustomerServiceAgent, load_config, load_prompts

    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    agent = CustomerServiceAgent(config, prompts)

    print("多模态客服智能体已启动，输入 'exit' 退出。\n")
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        text, image_ids = agent.answer(
            question=user_input,
            enable_cot=args.cot,
            enable_hallucination_check=args.hallucination_check,
        )
        print(f"\n客服: {text}")
        if image_ids:
            print(f"配图: {json.dumps(image_ids, ensure_ascii=False)}")
        print()


def cmd_eval(args):
    """批量评测模式"""
    os.system(
        f"python eval/eval_runner.py "
        f"--config {args.config} --prompts {args.prompts} "
        f"--questions {args.questions} --output {args.output} "
        f"--submission {args.submission}"
    )


def main():
    parser = argparse.ArgumentParser(description="多模态客服智能体")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--prompts", default="configs/prompts.yaml")

    subparsers = parser.add_subparsers(dest="command")

    # 预处理
    sub_pre = subparsers.add_parser("preprocess", help="数据预处理")

    # 交互对话
    sub_chat = subparsers.add_parser("chat", help="交互式对话")
    sub_chat.add_argument("--cot", action="store_true", help="启用思维链拆解")
    sub_chat.add_argument("--hallucination-check", action="store_true", help="启用幻觉检查")

    # 批量评测
    sub_eval = subparsers.add_parser("eval", help="批量评测")
    sub_eval.add_argument("--questions", default="eval/questions/questions_A.jsonl")
    sub_eval.add_argument("--output", default="eval/results/results_A.jsonl")
    sub_eval.add_argument("--submission", default="submissions/answer_A.json")

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
