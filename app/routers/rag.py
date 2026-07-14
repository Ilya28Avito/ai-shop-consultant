from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag import RAGService

router = APIRouter(prefix="/rag", tags=["rag"])

_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        _rag_service.build()
    return _rag_service


class RAGQuery(BaseModel):
    question: str


@router.post("/query", summary="RAG-запрос к базе знаний")
async def rag_query(body: RAGQuery):
    """Семантический поиск + генерация ответа через LlamaIndex."""
    service = get_rag_service()
    result = service.answer(body.question)
    return result
