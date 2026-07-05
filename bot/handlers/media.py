from io import BytesIO
from aiogram import Router, F
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from uuid import UUID

router = Router()


async def stream_response(message: Message, backend, chat_id: UUID, content: str,
                           media: bytes | None = None, mime: str | None = None):
    """Общая функция для стриминга ответа."""
    sent = await message.answer("⏳...")
    buffer = ""
    try:
        async for chunk in backend.send_message(chat_id, content, media=media, mime=mime):
            buffer += chunk
            try:
                await sent.edit_text(buffer)
            except Exception:
                pass
        if not buffer:
            await sent.edit_text("❌ Пустой ответ от сервиса")
    except Exception as e:
        await sent.edit_text(f"❌ Ошибка: {e}")


@router.message(F.photo)
async def handle_photo(message: Message, state: FSMContext, backend=None):
    """Обработка фотографий."""
    data = await state.get_data()
    chat_id = data.get("chat_id")

    if not chat_id:
        await message.answer("❌ Сначала запустите бота командой /start")
        return

    # Берём фото максимального размера
    photo = message.photo[-1]
    file = await message.bot.get_file(photo.file_id)
    buf = BytesIO()
    await message.bot.download_file(file.file_path, destination=buf)
    photo_bytes = buf.getvalue()

    caption = message.caption or "Что изображено на этом фото? Опиши подробно."

    await stream_response(
        message, backend, UUID(chat_id),
        content=caption,
        media=photo_bytes,
        mime="image/jpeg"
    )


@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext, backend=None):
    """Обработка голосовых сообщений."""
    data = await state.get_data()
    chat_id = data.get("chat_id")

    if not chat_id:
        await message.answer("❌ Сначала запустите бота командой /start")
        return

    file = await message.bot.get_file(message.voice.file_id)
    buf = BytesIO()
    await message.bot.download_file(file.file_path, destination=buf)
    voice_bytes = buf.getvalue()

    await stream_response(
        message, backend, UUID(chat_id),
        content="[голосовое сообщение]",
        media=voice_bytes,
        mime="audio/ogg"
    )


@router.message(F.document)
async def handle_document(message: Message, state: FSMContext, backend=None):
    """Обработка документов (PDF, DOCX)."""
    data = await state.get_data()
    chat_id = data.get("chat_id")

    if not chat_id:
        await message.answer("❌ Сначала запустите бота командой /start")
        return

    doc = message.document
    if not doc.file_name:
        await message.answer("❌ Документ без имени не поддерживается")
        return

    name_lower = doc.file_name.lower()
    if not (name_lower.endswith(".pdf") or name_lower.endswith(".docx")):
        await message.answer("❌ Поддерживаются только PDF и DOCX файлы")
        return

    if doc.file_size > 10 * 1024 * 1024:
        await message.answer("❌ Файл слишком большой (максимум 10 МБ)")
        return

    file = await message.bot.get_file(doc.file_id)
    buf = BytesIO()
    await message.bot.download_file(file.file_path, destination=buf)
    doc_bytes = buf.getvalue()

    caption = message.caption or "Проанализируй этот документ и кратко опиши его содержание."

    await stream_response(
        message, backend, UUID(chat_id),
        content=caption,
        media=doc_bytes,
        mime=doc.mime_type
    )
