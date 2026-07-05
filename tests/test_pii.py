from app.observability.pii import redact_pii


def test_email_redacted():
    text = "Мой email ivan@mail.ru, свяжитесь со мной"
    result = redact_pii(text)
    assert "ivan@mail.ru" not in result
    assert "[EMAIL]" in result


def test_phone_redacted():
    text = "Позвоните мне +7 (999) 123-45-67"
    result = redact_pii(text)
    assert "+7 (999) 123-45-67" not in result
    assert "[PHONE_RU]" in result


def test_card_redacted():
    text = "Карта 4111 1111 1111 1111"
    result = redact_pii(text)
    assert "4111 1111 1111 1111" not in result
    assert "[CARD]" in result


def test_multiple_pii_redacted():
    text = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111"
    result = redact_pii(text)
    assert "ivan@mail.ru" not in result
    assert "+7 (999) 123-45-67" not in result
    assert "4111 1111 1111 1111" not in result
    assert "[EMAIL]" in result
    assert "[PHONE_RU]" in result
    assert "[CARD]" in result


def test_clean_text_unchanged():
    text = "Хочу купить iPhone 15, есть ли в наличии?"
    result = redact_pii(text)
    assert result == text
