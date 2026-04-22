"""
本地评测指标：模拟 LLM 裁判，按赛题 5 分制对回答进行自评。
"""
import os
import json
import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """你是一位严格的评分裁判。请根据以下维度为客服智能体的回答打分（1-5分）。

评分标准：
1 分（质量差）：回答未回应问题，结构混乱或缺失，图片无关或无帮助。
2 分（质量一般）：回答部分回应问题但不完整；结构较弱，图文结合较差。
3 分（质量中等）：回答回应了问题但缺乏深度；结构清晰但可优化；图片有一定帮助。
4 分（质量良好）：回答清晰、较为全面；结构逻辑清晰，图片有助于理解文本。
5 分（质量优秀）：回答详细、有深度；结构严谨连贯，图片与文本完美互补。

【用户问题】：
{question}

【智能体回答】：
{answer}

请严格按照以上标准打分，输出格式为：
分数：X
理由：（简要说明打分原因）"""
)


def judge_single(question: str, answer: str, judge_llm) -> Dict:
    """用 LLM 对单条回答打分"""
    chain = JUDGE_PROMPT | judge_llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer})

    score = 3
    reasoning = result
    score_match = re.search(r"分数[：:]\s*(\d)", result)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(5, score))

    reason_match = re.search(r"理由[：:]\s*(.+)", result, re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return {"score": score, "reasoning": reasoning}


def _normalize_id(raw_id) -> str:
    return str(raw_id).strip()


def load_questions_map(questions_path: str) -> Dict[str, str]:
    """加载题目文件，返回 id -> question。支持 CSV / JSONL。"""
    suffix = Path(questions_path).suffix.lower()
    questions_map = {}

    if suffix == ".csv":
        with open(questions_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = _normalize_id(row.get("id", ""))
                question = (row.get("question") or "").strip()
                if qid:
                    questions_map[qid] = question
        return questions_map

    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = _normalize_id(row.get("id", ""))
            question = (row.get("question") or "").strip()
            if qid:
                questions_map[qid] = question
    return questions_map


def load_answer_records(results_path: str) -> List[Dict]:
    """
    加载待评测答案文件。
    支持:
    - results jsonl: 含 question + answer
    - submission csv: id + ret/answer
    - submission json/jsonl: id + answer
    """
    suffix = Path(results_path).suffix.lower()

    if suffix == ".csv":
        records = []
        with open(results_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "id": _normalize_id(row.get("id", "")),
                    "question": (row.get("question") or "").strip(),
                    "answer": (row.get("ret") or row.get("answer") or "").strip(),
                })
        return records

    if suffix == ".json":
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = []
        for row in data:
            records.append({
                "id": _normalize_id(row.get("id", "")),
                "question": (row.get("question") or "").strip(),
                "answer": (row.get("ret") or row.get("answer") or "").strip(),
            })
        return records

    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append({
                "id": _normalize_id(row.get("id", "")),
                "question": (row.get("question") or "").strip(),
                "answer": (row.get("ret") or row.get("answer") or "").strip(),
            })
    return records


def evaluate_results(
    results_path: str,
    output_path: str,
    judge_llm,
    questions_path: Optional[str] = None,
) -> Dict:
    """批量评测结果文件或提交文件。"""
    results = load_answer_records(results_path)
    questions_map = load_questions_map(questions_path) if questions_path else {}

    evaluated = []
    total_score = 0

    for i, r in enumerate(results):
        qid = _normalize_id(r.get("id", i + 1))
        question = r.get("question", "").strip() or questions_map.get(qid, "")
        answer = r.get("answer", "").strip()

        if not question:
            raise ValueError(
                f"题目 {qid} 缺少 question，且未能从 --questions 提供的题库中补全。"
            )

        print(f"[{i+1}/{len(results)}] 评测题目 {qid}...")
        judgment = judge_single(question, answer, judge_llm)
        entry = {"id": qid, "question": question, "answer": answer, **judgment}
        evaluated.append(entry)
        total_score += judgment["score"]

    avg_score = total_score / max(len(evaluated), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in evaluated:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    score_dist = {s: 0 for s in range(1, 6)}
    for e in evaluated:
        score_dist[e["score"]] = score_dist.get(e["score"], 0) + 1

    summary = {
        "total_questions": len(evaluated),
        "average_score": round(avg_score, 2),
        "score_distribution": score_dist,
    }

    print(f"\n评测完成：共 {len(evaluated)} 题，平均分 {avg_score:.2f}")
    print(f"分数分布: {score_dist}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM 裁判评测")
    parser.add_argument("--results", default="eval/results/results_A.jsonl")
    parser.add_argument("--output", default="eval/results/evaluated_A.jsonl")
    parser.add_argument("--questions", default=None, help="当 results 不含 question 时，用于补全题目文本")
    parser.add_argument("--model", default="kimi-k2-0905-preview")
    parser.add_argument("--provider", default="openai")
    args = parser.parse_args()

    prov = (args.provider or "openai").strip().lower()
    if prov == "google_genai":
        judge_llm = init_chat_model(
            args.model,
            model_provider=args.provider,
            api_key=os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
        )
    else:
        judge_llm = init_chat_model(
            args.model,
            model_provider=args.provider,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "").strip() or None,
            temperature=0.0,
        )
    evaluate_results(args.results, args.output, judge_llm, questions_path=args.questions)


if __name__ == "__main__":
    main()
