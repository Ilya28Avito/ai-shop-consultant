import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_bot_settings
from bot.services.backend_client import BackendClient
from bot.handlers import commands, text, fsm
from bot.handlers import media

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    settings = get_bot_settings()

    bot = Bot(token=settings.bot_token)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)

    # Создаём backend клиент
    backend = BackendClient(base_url=settings.backend_url)
    dp["backend"] = backend

    # Регистрируем роутеры
    dp.include_router(commands.router)
    dp.include_router(fsm.router)
    dp.include_router(media.router)
    dp.include_router(text.router)

    logger.info("Бот запускается...")
    try:
        await dp.start_polling(bot)
    finally:
        await backend.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
