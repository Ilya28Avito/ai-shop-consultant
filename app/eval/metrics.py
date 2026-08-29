"""
ДЗ 5.6 — модуль метрик для оценки RAG (используется scripts/run_eval.py).

Судья и продовая LLM — РАЗНЫЕ роли, не путать:
  - судья (эта модель) оценивает качество ответов через ragas.metrics.collections;
  - продовая LLM (см. app/services/rag.py, захардкожена как "gpt-4o-mini" в
    openai_client.chat.completions.create) генерирует сами ответы бота.
Сейчас у обеих ролей случайно одна и та же модель (gpt-4o-mini) — это
совпадение конфигурации, а не архитектурная связь: поменять одну модель
можно не трогая другую.

Судья — gpt-4o-mini через provider="openai" (llm_factory), а не
claude-sonnet-4-6/anthropic, как в дефолте методички: у проекта уже есть
OPENAI_API_KEY и осознанно не заводится Anthropic-ключ (см.
requirements-eval.txt). Модель и провайдер судьи можно переопределить через
переменные окружения EVAL_JUDGE_MODEL / EVAL_JUDGE_PROVIDER в .env_robust_23,
если это когда-нибудь понадобится поменять.
"""
import os

import openai
from pydantic import BaseModel, Field
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import discrete_metric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.metrics.result import MetricResult

DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "openai")
DEFAULT_JUDGE_EMBEDDING_MODEL = os.getenv("EVAL_JUDGE_EMBEDDING_MODEL", "text-embedding-3-small")


def build_judge(model: str = DEFAULT_JUDGE_MODEL, provider: str = DEFAULT_JUDGE_PROVIDER):
    """Судья для всех RAGAS-метрик. provider="openai" -> используем уже
    имеющийся OPENAI_API_KEY, отдельный клиент под Anthropic не заводим.

    ВАЖНО: клиент именно AsyncOpenAI, а не OpenAI. ascore() у метрик из
    ragas.metrics.collections — асинхронный метод (вызывает под капотом
    llm.agenerate()); с синхронным клиентом падает с TypeError
    "Cannot use agenerate() with a synchronous client" на каждой метрике.
    Это отличается от generate_testset.py, где TestsetGenerator использует
    другой, не async-only путь генерации — там синхронный клиент был ок."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY не найден — проверь .env_robust_23")
    client = openai.AsyncOpenAI(api_key=api_key)
    return llm_factory(model, provider=provider, client=client)


def build_embeddings(model: str = DEFAULT_JUDGE_EMBEDDING_MODEL):
    """Эмбеддинги нужны только AnswerRelevancy (сравнивает embedding вопроса
    с embedding вопросов, сгенерированных по ответу). Тоже AsyncOpenAI —
    по той же причине, что и в build_judge()."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY не найден — проверь .env_robust_23")
    client = openai.AsyncOpenAI(api_key=api_key)
    return OpenAIEmbeddings(client=client, model=model)


class _HasCitationVerdict(BaseModel):
    """Структурированный ответ судьи для has_citation — вместо парсинга
    свободного текста судья обязан вернуть эти два строго типизированных
    поля (instructor следит, что модель не может ответить чем-то другим)."""
    has_citation: bool = Field(
        description=(
            "True, если ответ содержит ссылку на источник: маркер вида "
            "'[1]'/'[doc_id]', имя файла источника, или фразу вида "
            "'согласно ...'/'по данным ...'. False, если ответ ничего "
            "подобного не содержит (в т.ч. если это отказ вида "
            "'по базе не нашёл' — там цитировать нечего)."
        )
    )
    reasoning: str = Field(description="Краткое обоснование в одно предложение, на русском")


def make_has_citation(judge):
    """ДЗ 5.6, шаг 4 — кастомная метрика has_citation через @discrete_metric.

    В RAGAS 0.3 для такого использовался AspectCritic, в 0.4 его убрали —
    @discrete_metric ему на замену (см. app/eval/metrics.py module docstring).
    Судья захватывается через замыкание (а не как параметр ascore) — иначе
    он попал бы в pydantic-схему аргументов метрики, которую @discrete_metric
    строит из сигнатуры функции.

    Проверяет только ФОРМАЛЬНОЕ наличие ссылки на источник в тексте ответа
    (маркер/имя файла/"согласно ..."), а не то, правильная ли это ссылка —
    это ровно критерий 3 из чекпоинта 5, который иначе пришлось бы проверять
    руками на каждом ответе.
    """
    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> MetricResult:
        prompt = (
            "Содержит ли следующий ответ ссылку на источник: маркер вида "
            "'[1]'/'[doc_id]', имя файла, или фразу 'согласно ...'?\n\n"
            f"Ответ бота:\n{response}"
        )
        verdict = await judge.agenerate(prompt, response_model=_HasCitationVerdict)
        return MetricResult(
            value="yes" if verdict.has_citation else "no",
            reason=verdict.reasoning,
        )

    return has_citation


def build_metrics(judge, embeddings) -> dict:
    """Пять метрик ДЗ 5.6: четыре из ragas.metrics.collections (шаг 3) +
    has_citation (шаг 4, через @discrete_metric)."""
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "has_citation": make_has_citation(judge),
    }


async def eval_row(
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
    metrics: dict,
) -> dict:
    """Считает все метрики для одной строки golden dataset.

    У разных метрик ragas.metrics.collections разный набор именованных
    аргументов ascore(...) (см. app/services/rag.py evaluate_inputs для
    контракта retrieved_contexts) — этот метод скрывает разницу, чтобы
    scripts/run_eval.py не знал деталей каждой метрики.

    Каждая метрика считается в своём try/except: если судья споткнулся на
    одной метрике (сетевой сбой, rate limit) — строка не теряется целиком,
    остальные метрики всё равно посчитаются, а сбойная просто уйдёт как NaN
    с текстом ошибки в *_error.
    """
    result: dict = {}

    async def _score(name: str, coro):
        try:
            metric_result = await coro
            result[name] = metric_result.value
            result[f"{name}_reason"] = getattr(metric_result, "reason", None)
        except Exception as e:
            result[name] = float("nan")
            result[f"{name}_reason"] = None
            result[f"{name}_error"] = f"{type(e).__name__}: {e}"

    await _score(
        "faithfulness",
        metrics["faithfulness"].ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        ),
    )
    await _score(
        "answer_relevancy",
        metrics["answer_relevancy"].ascore(
            user_input=user_input,
            response=response,
        ),
    )
    await _score(
        "context_precision",
        metrics["context_precision"].ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        ),
    )
    await _score(
        "context_recall",
        metrics["context_recall"].ascore(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        ),
    )
    if "has_citation" in metrics:
        await _score(
            "has_citation",
            metrics["has_citation"].ascore(response=response),
        )

    return result
