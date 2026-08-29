from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag import answer

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGQuery(BaseModel):
    question: str
    category: str | None = None


@router.post("/query", summary="RAG-запрос к базе знаний")
async def rag_query(body: RAGQuery):
    """Семантический поиск + генерация ответа с цитатами."""
    result = await answer(body.question, category=body.category)
    return result