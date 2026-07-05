from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext

from bot.states import AskFlow
from bot.keyboards.inline import topics_kb

router = Router()


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext):
    """Запуск FSM-сценария выбора темы."""
    await state.set_state(AskFlow.waiting_for_topic)
    await message.answer(
        "📋 Выберите тему вопроса:",
        reply_markup=topics_kb()
    )


@router.callback_query(AskFlow.waiting_for_topic, F.data.startswith("topic:"))
async def process_topic(callback: CallbackQuery, state: FSMContext):
    """Обработка выбора темы."""
    topic_slug = callback.data.split(":")[1]

    if topic_slug == "cancel":
        await state.clear()
        await callback.message.edit_text("❌ Отменено.")
        await callback.answer()
        return

    topic_names = {
        "delivery": "Доставка",
        "return": "Возврат товара",
        "payment": "Оплата",
        "warranty": "Гарантия",
        "support": "Поддержка",
    }
    topic_name = topic_names.get(topic_slug, topic_slug)

    await state.update_data(topic=topic_name)
    await state.set_state(AskFlow.waiting_for_question)
    await callback.message.edit_text(
        f"✅ Тема: {topic_name}\n\nТеперь напишите ваш вопрос:"
    )
    await callback.answer()


@router.message(AskFlow.waiting_for_question)
async def process_question(message: Message, state: FSMContext, backend=None):
    """Обработка вопроса в выбранной теме."""
    data = await state.get_data()
    topic = data.get("topic", "Общий")
    chat_id = data.get("chat_id")

    if not chat_id:
        await state.clear()
        await message.answer("❌ Сначала запустите бота командой /start")
        return

    prompt = f"Тема: {topic}. Вопрос: {message.text}"

    sent = await message.answer("⏳ Обрабатываю вопрос...")
    buffer = ""

    try:
        from uuid import UUID
        async for chunk in backend.send_message(UUID(chat_id), prompt):
            buffer += chunk
            try:
                await sent.edit_text(buffer)
            except Exception:
                pass
    except Exception as e:
        await sent.edit_text(f"❌ Ошибка: {e}")

    await state.clear()
