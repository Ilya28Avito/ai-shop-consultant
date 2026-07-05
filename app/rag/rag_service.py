import os
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from openai import AsyncOpenAI
from app.rag.retriever import retrieve

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAG_SYSTEM_PROMPT = """Ты — ИИ-консультант интернет-магазина ТехноМаркет.

Отвечай ТОЛЬКО на основе предоставленного контекста из базы знаний магазина.
Если информации нет в контексте — честно скажи что не знаешь.
Отвечай кратко — не более 3-4 предложений.

Контекст из базы знаний:
{context}"""


async def rag_chat(question: str) -> str:
    """Отвечает на вопрос используя RAG."""

    # 1. Находим релевантные документы
    chunks = await retrieve(question, k=3)
    context = "\n\n---\n\n".join(chunks)

    # 2. Формируем промпт с контекстом
    system = RAG_SYSTEM_PROMPT.format(context=context)

    # 3. Запрос к LLM
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


async def rag_chat_with_sources(question: str) -> tuple[str, list[str]]:
    """Отвечает на вопрос и возвращает источники."""
    from app.rag.retriever import retrieve_with_scores

    results = await retrieve_with_scores(question, k=3)
    chunks = [chunk for chunk, score in results]
    scores = [score for chunk, score in results]
    context = "\n\n---\n\n".join(chunks)

    system = RAG_SYSTEM_PROMPT.format(context=context)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content
    return answer, chunks
