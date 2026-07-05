import asyncio
import json
import os
import sys
import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(".env_robust_23")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """Ты — эксперт по оценке качества ответов ИИ-ассистента интернет-магазина.

Оцени ответ ассистента по трём критериям (1-5):
- relevance: насколько ответ соответствует вопросу
- correctness: насколько ответ правильный и точный
- completeness: насколько ответ полный

Вопрос: {question}
Ожидаемый ответ: {expected_answer}
Ключевые слова: {keywords}
Фактический ответ: {actual_answer}

Сначала напиши reasoning (рассуждение), потом выставь оценки.
Верни JSON строго в формате:
{{
  "reasoning": "твоё рассуждение здесь",
  "relevance": 4,
  "correctness": 4,
  "completeness": 3,
  "explanation": "краткое объяснение"
}}"""


async def get_answer(question: str, model: str = "gpt-4o-mini") -> str:
    """Получаем ответ от нашего консультанта."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты ИИ-консультант интернет-магазина ТехноМаркет. Отвечай кратко и по делу."},
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    return response.choices[0].message.content


async def judge_answer(question: str, expected: str, keywords: list, actual: str, judge_model: str) -> dict:
    """LLM-as-judge оценивает ответ."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected,
        keywords=", ".join(keywords),
        actual_answer=actual,
    )
    response = await client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def run_evaluation(golden_path: str, judge_model: str, out_path: str):
    """Основной цикл оценки."""
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    items = golden["items"]
    results = []

    print(f"Оцениваем {len(items)} вопросов...")

    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {item['id']}: {item['question'][:50]}...")

        # Получаем ответ консультанта
        actual = await get_answer(item["question"])

        # Оцениваем судьёй
        scores = await judge_answer(
            question=item["question"],
            expected=item["expected_answer"],
            keywords=item.get("expected_keywords", []),
            actual=actual,
            judge_model=judge_model,
        )

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": actual,
            "scores": {
                "relevance": scores.get("relevance", 0),
                "correctness": scores.get("correctness", 0),
                "completeness": scores.get("completeness", 0),
            },
            "reasoning": scores.get("reasoning", ""),
            "explanation": scores.get("explanation", ""),
        })

    # Считаем агрегаты
    relevance_avg = sum(r["scores"]["relevance"] for r in results) / len(results)
    correctness_avg = sum(r["scores"]["correctness"] for r in results) / len(results)
    completeness_avg = sum(r["scores"]["completeness"] for r in results) / len(results)
    min_correctness = min(r["scores"]["correctness"] for r in results)

    output = {
        "run_id": f"run_{date.today()}",
        "timestamp": str(date.today()),
        "model_under_test": "gpt-4o-mini",
        "judge_model": judge_model,
        "golden_version": golden.get("version", 1),
        "items": results,
        "aggregates": {
            "relevance_avg": round(relevance_avg, 2),
            "correctness_avg": round(correctness_avg, 2),
            "completeness_avg": round(completeness_avg, 2),
            "min_correctness": min_correctness,
        }
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены: {out_path}")
    print(f"   relevance_avg:    {relevance_avg:.2f}")
    print(f"   correctness_avg:  {correctness_avg:.2f}")
    print(f"   completeness_avg: {completeness_avg:.2f}")
    print(f"   min_correctness:  {min_correctness}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_dataset.json")
    parser.add_argument("--judge", default="gpt-4o-mini")
    parser.add_argument("--out", default=f"eval/runs/{date.today()}.json")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.golden, args.judge, args.out))
