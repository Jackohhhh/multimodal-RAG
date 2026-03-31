"""
批量答题脚本：读入赛题，调用 Agent 逐题生成回答，保存结果。
"""
import os
import json
import argparse
import time
import yaml
from typing import List, Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.agent import CustomerServiceAgent, load_config, load_prompts


def load_questions(questions_path: str) -> List[Dict]:
    """
    加载赛题文件。支持 JSONL 格式，每行一个 JSON 对象。
    必须包含 'id' 和 'question' 字段，可选 'image_path' 字段。
    """
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj)
    print(f"加载了 {len(questions)} 道赛题")
    return questions


def run_evaluation(
    agent: CustomerServiceAgent,
    questions: List[Dict],
    output_path: str,
    enable_cot: bool = True,
    enable_hallucination_check: bool = True,
):
    """批量运行 Agent 回答，逐条保存结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    for i, q in enumerate(questions):
        qid = q.get("id", i + 1)
        question_text = q.get("question", "")
        image_path = q.get("image_path")

        print(f"\n[{i+1}/{len(questions)}] 题目 {qid}: {question_text[:60]}...")
        start_time = time.time()

        try:
            answer_text, image_ids = agent.answer(
                question=question_text,
                image_path=image_path,
                enable_cot=enable_cot,
                enable_hallucination_check=enable_hallucination_check,
            )
        except Exception as e:
            answer_text = f"回答生成失败: {e}"
            image_ids = []
            print(f"  错误: {e}")

        elapsed = time.time() - start_time
        result = {
            "id": qid,
            "question": question_text,
            "answer": answer_text,
            "image_ids": image_ids,
            "elapsed_seconds": round(elapsed, 2),
        }
        results.append(result)

        print(f"  耗时 {elapsed:.1f}s，回答长度 {len(answer_text)} 字，配图 {len(image_ids)} 张")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n全部完成！结果保存到 {output_path}")
    return results


def export_submission(results: List[Dict], submission_path: str):
    """将结果导出为赛题提交格式"""
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission = []
    for r in results:
        submission.append({
            "id": r["id"],
            "answer": r["answer"],
            "image_ids": r.get("image_ids", []),
        })
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"提交文件已导出到 {submission_path}")


def main():
    parser = argparse.ArgumentParser(description="批量答题评测")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--prompts", default="configs/prompts.yaml")
    parser.add_argument("--questions", default="eval/questions/questions_A.jsonl")
    parser.add_argument("--output", default="eval/results/results_A.jsonl")
    parser.add_argument("--submission", default="submissions/answer_A.json")
    parser.add_argument("--no-cot", action="store_true", help="禁用思维链拆解")
    parser.add_argument("--no-hallucination-check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    agent = CustomerServiceAgent(config, prompts)

    questions = load_questions(args.questions)
    results = run_evaluation(
        agent=agent,
        questions=questions,
        output_path=args.output,
        enable_cot=not args.no_cot,
        enable_hallucination_check=not args.no_hallucination_check,
    )
    export_submission(results, args.submission)


if __name__ == "__main__":
    main()
