import pytest
from pydantic import ValidationError
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage
from app.observability.pii import redact_pii


def test_chat_request_valid():
    req = ChatRequest(messages=[Message(role="user", content="Привет")])
    assert req.temperature == 0.7
    assert req.max_tokens == 1000


def test_chat_request_temperature_validation():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[Message(role="user", content="Привет")],
            temperature=3.0
        )


def test_chat_request_negative_temperature():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[Message(role="user", content="Привет")],
            temperature=-1.0
        )


def test_chat_request_max_tokens_validation():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[Message(role="user", content="Привет")],
            max_tokens=99999
        )


def test_chat_request_zero_max_tokens():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[Message(role="user", content="Привет")],
            max_tokens=0
        )


def test_chat_response_cached_false_by_default():
    resp = ChatResponse(
        content="Ответ",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        finish_reason="stop",
    )
    assert resp.cached is False


def test_pii_in_message_redacted():
    text = "Мой email test@example.com и телефон +7 999 123 45 67"
    result = redact_pii(text)
    assert "test@example.com" not in result
    assert "[EMAIL]" in result


def test_message_roles():
    for role in ["user", "assistant", "system"]:
        msg = Message(role=role, content="Текст")
        assert msg.role == role
