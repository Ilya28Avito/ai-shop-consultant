import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

from app.prompts.loader import render_system_prompt
from app.tools.schemas import TOOLS
from app.tools.handlers import execute_tool

load_dotenv(".env_robust_23")

# ============================
# ЛОГИРОВАНИЕ
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_with_tools(user_input: str, shop_name: str = "ТехноМаркет") -> str:
    """
    Полный цикл Function Calling:
    1. Отправляем вопрос + инструменты
    2. Если модель вызвала tool → выполняем функцию
    3. Отправляем результат обратно
    4. Получаем финальный ответ
    """

    # Загружаем system prompt из файла
    system_prompt = render_system_prompt(version="v1", shop_name=shop_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    logger.info(f"INPUT: {user_input}")

    # ШАГ 1: первый запрос к модели
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message

    # ШАГ 2: проверяем — модель вызвала tool или ответила текстом?
    if not message.tool_calls:
        # Модель ответила текстом — tool не нужен
        logger.info(f"NO TOOL CALLED | ANSWER: {message.content}")
        logger.info(f"TOKENS: {response.usage.total_tokens}")
        return message.content

    # ШАГ 3: модель вызвала tool — выполняем функцию
    # Добавляем ответ ассистента с tool_calls в историю
    messages.append(message)

    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        logger.info(f"TOOL CALLED: {tool_name} | ARGS: {tool_args}")

        # Выполняем функцию
        tool_result = execute_tool(tool_name, tool_args)

        logger.info(f"TOOL RESULT: {tool_result}")

        # Добавляем результат в историю
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

    # ШАГ 4: второй запрос — модель формулирует финальный ответ
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    final_answer = final_response.choices[0].message.content
    total_tokens = response.usage.total_tokens + final_response.usage.total_tokens

    logger.info(f"FINAL ANSWER: {final_answer}")
    logger.info(f"TOTAL TOKENS: {total_tokens}")

    return final_answer
