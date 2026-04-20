"""
批量答题脚本：读入赛题，调用 Agent 逐题生成回答，保存结果。
"""
import os
import json
import argparse
import time
import csv
import re
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.agent import CustomerServiceAgent, load_config, load_prompts


DEFAULT_QUESTIONS_PATH = "eval/questions/questions_A.jsonl"
DEFAULT_OUTPUT_PATH = "eval/results/results_A.jsonl"
DEFAULT_SUBMISSION_PATH = "submissions/answer_A.json"


def normalize_question_text(question: str) -> str:
    """清洗 CSV 中多行/多重引号包裹的问题文本。"""
    normalized = (question or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned_lines = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r'^[\"“”]+', "", line)
        line = re.sub(r'[\"“”]+[,，]?$', "", line)
        line = re.sub(r'[,，]$', "", line)
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def normalize_submission_text(answer: str) -> str:
    """将提交答案压成单段文本，避免换行分段直接写入 CSV。"""
    text = (answer or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_submission_ret(answer: str, image_ids: List[str]) -> str:
    """
    按提交要求组织 ret 字段：
    - 有图: "回答文本",["img1","img2"]
    - 无图: "回答文本"
    """
    normalized_answer = normalize_submission_text(answer)
    answer_part = json.dumps(normalized_answer, ensure_ascii=False)
    if not image_ids:
        return answer_part
    image_ids_part = json.dumps(image_ids, ensure_ascii=False)
    return f"{answer_part},{image_ids_part}"


def load_questions_from_jsonl(questions_path: str) -> List[Dict]:
    """加载 JSONL 题目文件。"""
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj)
    return questions


def load_questions_from_csv(questions_path: str) -> List[Dict]:
    """加载 CSV 题目文件。至少需要 id/question 两列。"""
    questions = []
    with open(questions_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            normalized_row = {(key or "").strip(): value for key, value in row.items()}
            qid = normalized_row.get("id") or idx
            question_text = normalize_question_text(normalized_row.get("question", ""))
            image_path = (normalized_row.get("image_path") or "").strip() or None
            questions.append(
                {
                    "id": qid,
                    "question": question_text,
                    "image_path": image_path,
                }
            )
    return questions


def load_questions(questions_path: str) -> List[Dict]:
    """
    加载赛题文件。支持 JSONL / CSV。
    必须包含 'id' 和 'question' 字段，可选 'image_path' 字段。
    """
    suffix = Path(questions_path).suffix.lower()
    if suffix == ".csv":
        questions = load_questions_from_csv(questions_path)
    else:
        questions = load_questions_from_jsonl(questions_path)
    print(f"加载了 {len(questions)} 道赛题")
    return questions


def resolve_output_paths(
    questions_path: str,
    output_path: str,
    submission_path: str,
) -> tuple[str, str]:
    """
    如果用户没有显式指定输出文件，则根据题库文件名自动推导默认路径。
    例如 question_public.csv -> eval/results/question_public_results.jsonl
    """
    stem = Path(questions_path).stem
    questions_suffix = Path(questions_path).suffix.lower()
    if output_path == DEFAULT_OUTPUT_PATH:
        output_path = str(Path("eval/results") / f"{stem}_results.jsonl")
    if submission_path == DEFAULT_SUBMISSION_PATH:
        if questions_suffix == ".csv":
            submission_path = str(Path("submissions") / f"{stem}_submission.csv")
        else:
            submission_path = str(Path("submissions") / f"{stem}_submission.json")
    return output_path, submission_path


def run_evaluation(
    agent: CustomerServiceAgent,
    questions: List[Dict],
    output_path: str,
    enable_cot: bool = False,
    enable_hallucination_check: bool = False,
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
    """将结果导出为赛题提交格式，支持 JSON / JSONL / CSV。"""
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission = []
    for r in results:
        submission.append({
            "id": r["id"],
            "answer": normalize_submission_text(r["answer"]),
            "image_ids": r.get("image_ids", []),
        })

    suffix = Path(submission_path).suffix.lower()
    if suffix == ".jsonl":
        with open(submission_path, "w", encoding="utf-8") as f:
            for row in submission:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif suffix == ".csv":
        with open(submission_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "ret"])
            writer.writeheader()
            for row in submission:
                writer.writerow({
                    "id": row["id"],
                    "ret": format_submission_ret(row["answer"], row["image_ids"]),
                })
    else:
        with open(submission_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"提交文件已导出到 {submission_path}")


def main():
    parser = argparse.ArgumentParser(description="批量答题评测")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--prompts", default="configs/prompts.yaml")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--submission", default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--cot", action="store_true", help="启用思维链拆解")
    parser.add_argument("--hallucination-check", action="store_true", help="启用幻觉检查")
    args = parser.parse_args()

    args.output, args.submission = resolve_output_paths(
        questions_path=args.questions,
        output_path=args.output,
        submission_path=args.submission,
    )

    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    agent = CustomerServiceAgent(config, prompts)

    questions = load_questions(args.questions)
    results = run_evaluation(
        agent=agent,
        questions=questions,
        output_path=args.output,
        enable_cot=args.cot,
        enable_hallucination_check=args.hallucination_check,
    )
    export_submission(results, args.submission)


if __name__ == "__main__":
    main()
