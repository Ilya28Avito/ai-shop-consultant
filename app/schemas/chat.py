from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1, le=16000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "Есть ли iPhone 15 в наличии?"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            ]
        }
    }


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    finish_reason: str
    cached: bool = False

    @classmethod
    def from_openai(cls, response, cached: bool = False) -> "ChatResponse":
        """Адаптер из raw OpenAI ответа в нашу схему."""
        return cls(
            content=response.choices[0].message.content,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            finish_reason=response.choices[0].finish_reason,
            cached=cached,
        )


class ChatDelta(BaseModel):
    content: Optional[str] = None
    usage: Optional[Usage] = None
