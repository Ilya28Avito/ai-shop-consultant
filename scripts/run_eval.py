"""
ДЗ 5.6, шаг 3 — прогон golden dataset через прод-RAG и подсчёт RAGAS-метрик.

Что делает:
  1. Грузит tests/eval/golden_dataset.json (>=30 вычитанных руками пар).
  2. По каждому вопросу дёргает app.services.rag.evaluate_inputs(question) —
     ТОТ ЖЕ retrieval + score-guard + генерация, что и прод /rag/query,
     но с полным текстом найденных чанков вместо 200-симв. сниппета.
  3. Судья (по умолчанию gpt-4o-mini, provider=openai — см. app/eval/metrics.py)
     считает Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
     через ragas.metrics.collections.
  4. Сохраняет:
       tests/eval/results/{YYYY-MM-DD_HHMM}_{label}.csv   — per-row результат
       tests/eval/results/{YYYY-MM-DD_HHMM}_{label}.json  — агрегаты (среднее
       по каждой метрике, latency, число ошибок) — это audit-лог: по разным
       label (baseline, chunk_1024, top_k_10, ...) в шагах 6-7 строится
       таблица до/после в docs/rag_evaluation.md.

Запуск:
    python scripts/run_eval.py --label baseline

Параметры судьи (модель, эмбеддинги) можно переопределить через
EVAL_JUDGE_MODEL / EVAL_JUDGE_PROVIDER / EVAL_JUDGE_EMBEDDING_MODEL
в .env_robust_23 — по умолчанию не нужно, там уже разумные дефолты
(gpt-4o-mini / openai / text-embedding-3-small).

ВАЖНО: нужен поднятый Qdrant с уже проиндексированными данными (то, что
использует прод /rag/query) — если docker compose с qdrant не запущен,
скрипт упадёт на первом же вопросе с connection error.
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # python scripts/run_eval.py кладёт в sys.path только scripts/, а не корень
    # проекта — без этого `from app...` не найдётся, откуда бы ни запускали.
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(str(ROOT / ".env_robust_23"), override=True)

import pandas as pd

from app.eval.metrics import build_embeddings, build_judge, build_metrics, eval_row
from app.services.rag import evaluate_inputs

DEFAULT_GOLDEN = ROOT / "tests" / "eval" / "golden_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "eval" / "results"
DEFAULT_CONCURRENCY = 5


def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Golden dataset не найден: {path}\n"
            f"Сначала шаг 2: scripts/generate_testset.py + scripts/csv_to_golden_json.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) < 30:
        print(
            f"ВНИМАНИЕ: в golden dataset {len(data)} записей, меньше минимума 30 из задания. "
            f"Прогон всё равно пойдёт, но по критериям самопроверки это не пройдёт."
        )
    return data


async def eval_one(
    row: dict,
    metrics: dict,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict:
    """Полный цикл для одной golden-строки: retrieval+генерация прод-RAG,
    потом 4 RAGAS-метрики. Семафор ограничивает конкурентность — без него
    33 параллельных вопроса x несколько LLM-вызовов на метрику легко упрутся
    в rate limit OpenAI."""
    async with semaphore:
        started = time.monotonic()
        out = {
            "user_input": row["user_input"],
            "reference": row["reference"],
            "reference_contexts": row.get("reference_contexts", []),
        }
        rag_started = time.monotonic()
        try:
            rag_result = await evaluate_inputs(row["user_input"])
            # rag_latency_ms — время ТОЛЬКО retrieval+генерации, то есть то,
            # сколько реально ждёт пользователь в чате. Это и есть "latency"
            # для таблицы до/после в шагах 6-7. Общее время строки (ниже,
            # total_latency_ms) включает ещё и 4 вызова судьи поверх — это
            # расход самого eval-прогона, к продовой производительности
            # отношения не имеет, но полезно для оценки, сколько будет идти
            # A/B-эксперимент на N вопросов.
            out["rag_latency_ms"] = round((time.monotonic() - rag_started) * 1000)
            out["response"] = rag_result["response"]
            out["retrieved_contexts"] = rag_result["retrieved_contexts"]

            scores = await eval_row(
                user_input=row["user_input"],
                response=rag_result["response"],
                retrieved_contexts=rag_result["retrieved_contexts"],
                reference=row["reference"],
                metrics=metrics,
            )
            out.update(scores)
        except Exception as e:
            out.setdefault("rag_latency_ms", round((time.monotonic() - rag_started) * 1000))
            out["response"] = None
            out["retrieved_contexts"] = []
            out["row_error"] = f"{type(e).__name__}: {e}"

        out["total_latency_ms"] = round((time.monotonic() - started) * 1000)

        status = "OK" if "row_error" not in out else f"ОШИБКА: {out['row_error']}"
        print(f"[{index}/{total}] {status} | {row['user_input'][:70]}")
        return out


def aggregate(df: pd.DataFrame, label: str, judge_model: str, judge_provider: str,
              embedding_model: str, golden_path: Path) -> dict:
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    failed = int(df["row_error"].notna().sum()) if "row_error" in df.columns else 0

    means = {}
    for col in metric_cols:
        if col in df.columns:
            mean_val = df[col].astype(float).mean(skipna=True)
            means[col] = None if pd.isna(mean_val) else round(float(mean_val), 4)
        else:
            means[col] = None

    # has_citation — не число (строка "yes"/"no"), поэтому не строится через
    # .astype(float).mean() как остальные метрики: аггрегат — это доля "yes"
    # среди тех строк, где метрика вообще посчиталась (сбойные has_citation_error
    # строки исключаются из знаменателя, как и NaN у числовых метрик).
    has_citation_avg = None
    if "has_citation" in df.columns:
        scored = df["has_citation"].dropna()
        if len(scored):
            has_citation_avg = round(float((scored == "yes").mean()), 4)

    return {
        "label": label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "golden_dataset": str(golden_path),
        "n_rows": len(df),
        "n_failed_rows": failed,
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "judge_embedding_model": embedding_model,
        # avg_rag_latency_ms — это и есть "latency" для таблицы до/после
        # в отчёте (шаги 6-7): чистое время ответа прод-RAG, без судьи.
        "avg_rag_latency_ms": round(float(df["rag_latency_ms"].mean()), 1) if "rag_latency_ms" in df.columns else None,
        # avg_total_latency_ms — справочно: сколько в среднем идёт вся
        # строка целиком (RAG + 4 вызова судьи), пригодится только чтобы
        # прикинуть длительность будущих прогонов, в отчёт не идёт.
        "avg_total_latency_ms": round(float(df["total_latency_ms"].mean()), 1) if "total_latency_ms" in df.columns else None,
        "has_citation_avg": has_citation_avg,
        **{f"{k}_avg": v for k, v in means.items()},
    }


async def main(args: argparse.Namespace) -> None:
    golden = load_golden(args.golden)
    print(f"Загружено {len(golden)} пар из {args.golden}\n")

    judge = build_judge(model=args.judge_model, provider=args.judge_provider)
    embeddings = build_embeddings(model=args.judge_embedding_model)
    metrics = build_metrics(judge, embeddings)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        eval_one(row, metrics, semaphore, i, len(golden))
        for i, row in enumerate(golden, 1)
    ]

    print(f"Считаем метрики (судья: {args.judge_model}, конкурентность: {args.concurrency})...\n")
    rows = await asyncio.gather(*tasks)

    df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_path = args.output_dir / f"{timestamp}_{args.label}.csv"
    json_path = args.output_dir / f"{timestamp}_{args.label}.json"

    # utf-8-sig — та же история с Excel, что и в golden dataset: без BOM
    # кириллица в CSV на Windows превращается в мохибейк при обычном открытии.
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    agg = aggregate(
        df, args.label, args.judge_model, args.judge_provider,
        args.judge_embedding_model, args.golden,
    )
    json_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nСохранено:\n  {csv_path}\n  {json_path}\n")
    print("Средние по метрикам:")
    for key in ["faithfulness_avg", "answer_relevancy_avg", "context_precision_avg", "context_recall_avg", "has_citation_avg"]:
        print(f"  {key}: {agg[key]}")
    print(f"  avg_rag_latency_ms (это \"latency\" для отчёта): {agg['avg_rag_latency_ms']}")
    print(f"  avg_total_latency_ms (справочно, RAG + судья): {agg['avg_total_latency_ms']}")
    if agg["n_failed_rows"]:
        print(f"\nВНИМАНИЕ: {agg['n_failed_rows']} строк(и) упали с ошибкой — см. колонку row_error в CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ДЗ 5.6, шаг 3 — прогон RAGAS-метрик по golden dataset")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Путь к golden_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Куда сохранять CSV/JSON")
    parser.add_argument(
        "--label", type=str, default="baseline",
        help="Человекочитаемый label конфигурации (baseline, chunk_1024, top_k_10, ...) — идёт в имя файла",
    )
    parser.add_argument("--judge-model", type=str, default=None, help="Модель судьи (по умолчанию из .env/EVAL_JUDGE_MODEL или gpt-4o-mini)")
    parser.add_argument("--judge-provider", type=str, default=None, help="Провайдер судьи (по умолчанию openai)")
    parser.add_argument("--judge-embedding-model", type=str, default=None, help="Модель эмбеддингов судьи для AnswerRelevancy")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Сколько строк оценивать параллельно")
    args = parser.parse_args()

    # None -> дефолты по env берутся уже внутри build_judge/build_embeddings,
    # но argparse даёт возможность переопределить прямо из командной строки.
    from app.eval.metrics import DEFAULT_JUDGE_EMBEDDING_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROVIDER
    args.judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
    args.judge_provider = args.judge_provider or DEFAULT_JUDGE_PROVIDER
    args.judge_embedding_model = args.judge_embedding_model or DEFAULT_JUDGE_EMBEDDING_MODEL

    asyncio.run(main(args))
