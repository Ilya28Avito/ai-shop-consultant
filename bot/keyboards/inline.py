from aiogram.utils.keyboard import InlineKeyboardBuilder


def topics_kb():
    """Клавиатура с темами для ИИ-консультанта ТехноМаркет."""
    builder = InlineKeyboardBuilder()
    topics = [
        ("📦 Доставка", "topic:delivery"),
        ("↩️ Возврат товара", "topic:return"),
        ("💳 Оплата", "topic:payment"),
        ("🔧 Гарантия", "topic:warranty"),
        ("📞 Поддержка", "topic:support"),
    ]
    for text, callback_data in topics:
        builder.button(text=text, callback_data=callback_data)
    builder.button(text="❌ Отмена", callback_data="topic:cancel")
    builder.adjust(2)
    return builder.as_markup()
