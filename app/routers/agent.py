"""Блок 6.4 — SSE-стриминг персистентного графа-агента.

POST /agent/stream принимает {"thread_id": "...", "input": {...}}:
  - первый запрос по thread_id:  {"input": {"messages": [{"role": "user", "content": "..."}]}}
  - возобновление после interrupt: {"input": {"resume": true}} (или false)

Стримит через graph.astream(stream_mode=["updates", "messages"]):
  - "updates"  — прогресс по узлам графа: какой узел завершился и что вернул,
    плюс момент __interrupt__ (payload подтверждения, если граф остановился);
  - "messages" — токены LLM по мере генерации (для отзывчивого UI).

Не используем astream_events(v2): он даёт более гранулярные события
(on_chat_model_start/on_chat_model_stream/on_tool_start и т.п.) — удобно,
если нужен единый event-tree для сложного UI, но для двух категорий,
которые реально нужны здесь (прогресс узлов + токены модели), это
избыточная детализация и заметно больший объём трафика на SSE-соединении.
Обоснование — docs/agent-persistent-report.md, раздел 6.
"""
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from app.services.agent_persistent import build_initial_state

router = APIRouter(prefix="/agent", tags=["agent"])


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@router.post("/stream")
async def stream_agent(request: Request):
    body = await request.json()
    thread_id = body["thread_id"]
    payload = body.get("input", {})
    # ДЗ 6.4 — user_role: концепция уровня доступа, протянутая через
    # config["configurable"] (read_only / write_with_approve / full).
    # Сейчас confirm_and_execute_send_telegram роль не читает и interrupt()
    # вызывает всегда — это сознательное решение (см.
    # docs/agent-persistent-report.md, раздел 7): пропуск подтверждения для
    # "full" — бизнес-решение о доверии конкретным ролям, а не техническое,
    # поэтому здесь заведена только сама возможность передать роль по цепочке
    # вызовов, без готовой ветки автоапрува.
    user_role = body.get("user_role", "write_with_approve")
    config = {"configurable": {"thread_id": thread_id, "user_role": user_role}}
    agent = request.app.state.agent

    if "resume" in payload:
        graph_input = Command(resume=payload["resume"])
    else:
        # НАЙДЕННЫЙ ПРИ ОТЛАДКЕ БАГ (см. app/services/agent_persistent.py,
        # build_initial_state, и docs/agent-persistent-report.md раздел 7):
        # раньше state собирался вручную здесь же, без системного промпта —
        # граф работал вообще без SYSTEM_PROMPT, из-за чего HIL срабатывал
        # непоследовательно. Теперь единственный способ начать новый тред —
        # build_initial_state(), который явно подставляет системное сообщение.
        graph_input = build_initial_state(payload.get("messages", []))

    async def event_generator():
        async for stream_type, chunk in agent.astream(
            graph_input, config, stream_mode=["updates", "messages"]
        ):
            if stream_type == "updates":
                for node_name, node_update in chunk.items():
                    if node_name == "__interrupt__":
                        # node_update — кортеж Interrupt-объектов; в этом
                        # графе на паузу уходит ровно один interrupt() за
                        # раз, берём первый.
                        interrupt_obj = node_update[0]
                        yield _sse({"type": "interrupt", "payload": interrupt_obj.value})
                    else:
                        yield _sse({"type": "update", "node": node_name})
            elif stream_type == "messages":
                message_chunk, metadata = chunk
                content = getattr(message_chunk, "content", None)
                if content:
                    yield _sse({
                        "type": "token",
                        "node": metadata.get("langgraph_node"),
                        "content": content,
                    })

        yield _sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
