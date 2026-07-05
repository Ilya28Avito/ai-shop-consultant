from aiogram import Router, F
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from uuid import UUID

router = Router()


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, state: FSMContext, backend=None):
    """Обработка текстовых сообщений — отправка в backend."""
    data = await state.get_data()
    chat_id = data.get("chat_id")

    if not chat_id:
        await message.answer("❌ Сначала запустите бота командой /start")
        return

    sent = await message.answer("⏳...")
    buffer = ""

    try:
        async for chunk in backend.send_message(UUID(chat_id), message.text):
            buffer += chunk
            try:
                await sent.edit_text(buffer)
            except Exception:
                pass
        if not buffer:
            await sent.edit_text("❌ Пустой ответ от сервиса")
    except Exception as e:
        await sent.edit_text(f"❌ Ошибка: {e}")
