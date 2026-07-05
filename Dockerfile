# syntax=docker/dockerfile:1.7
FROM python:3.13-slim-bookworm AS builder

WORKDIR /app

# Копируем uv
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

# Переменные для uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Копируем зависимости ПЕРВЫМИ — для кеша слоёв
COPY requirements.txt .

# Устанавливаем зависимости
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Копируем код
COPY app/ ./app/

# ============================
# Runtime стадия
# ============================
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Создаём non-root пользователя
RUN useradd --create-home --uid 1000 appuser

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
