from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext, backend=None):
    """Приветствие и создание чата."""
    await state.clear()
    owner_id = str(message.from_user.id)

    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=owner_id,
            interface="telegram"
        )
        await state.update_data(chat_id=str(chat_id))
        await message.answer(
            "👋 Привет! Я ИИ-консультант магазина ТехноМаркет.\n\n"
            "Задайте любой вопрос о товарах, доставке или оплате.\n\n"
            "Команды:\n"
            "/ask — выбрать тему вопроса\n"
            "/clear — очистить историю\n"
            "/help — помощь\n"
            "/cancel — отменить текущее действие"
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка подключения к сервису: {e}")


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Справка по командам."""
    await message.answer(
        "📋 Доступные команды:\n\n"
        "/start — начать диалог\n"
        "/ask — задать вопрос по теме\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущее действие\n"
        "/help — показать эту справку\n\n"
        "Или просто напишите вопрос!"
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message, state: FSMContext, backend=None):
    """Очистка истории."""
    data = await state.get_data()
    chat_id = data.get("chat_id")

    if not chat_id:
        await message.answer("❌ Сначала запустите бота командой /start")
        return

    try:
        from uuid import UUID
        await backend.clear_messages(UUID(chat_id))
        await message.answer("✅ История диалога очищена!")
    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    """Отмена текущего действия."""
    await state.set_state(None)
    await message.answer("❌ Действие отменено.")
