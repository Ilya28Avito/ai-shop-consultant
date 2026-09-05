# ml-service — MLOps pipeline (credit scoring)

Учебный проект под ДЗ вебинара "MLOps без лишней сложности" (Академия EDPRO).
Структура повторяет методичку вебинара: DVC-пайплайн → MLflow (tracking +
model registry + алиасы) → FastAPI-сервис, отдающий предсказания по алиасу
модели → Evidently (мониторинг дрейфа) → CI в GitHub Actions.

## Структура

```
scripts/gen_train_data.py   генерация обучающей выборки (синтетика, seed=42)
scripts/gen_batch.py        генерация "продового" батча для мониторинга дрейфа
src/train.py                обучение модели, логирование в MLflow
params.yaml / dvc.yaml      DVC-пайплайн (стадия train)
app/main.py                 FastAPI: /health, /predict?alias=production|staging
monitor.py                  Evidently: отчёт по дрейфу + gate (exit 1 при превышении порога)
build_release_manifest.py   сборка release_manifest.json ("паспорт релиза")
Dockerfile                  образ для app/main.py
.github/workflows/ci.yml    CI: repro пайплайна, drift gate, docker build, smoke-test, публикация в GHCR (только main)
dvc.lock, metrics.json      реальный результат прогона пайплайна (roc_auc ≈ 0.90)
```

## Как воспроизвести локально

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install dvc

dvc init   # если ещё не инициализирован в этой папке
python scripts/gen_train_data.py

# в отдельном терминале — поднять MLflow tracking server:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

dvc repro
dvc metrics show
```

Дальше — зарегистрировать модель и выставить алиасы `production`/`staging`
через `MlflowClient.set_registered_model_alias(...)` (см. методичку), после
чего `app/main.py` сможет грузить модель по алиасу.

## CI vs CD

`.github/workflows/ci.yml` — это **CI** (continuous integration): каждый push
проверяет, что код + пайплайн + образ вообще собираются и работают
(repro, drift gate, smoke-test контейнера). Единственный шаг, который что-то
*публикует* — пуш образа в GHCR — выполняется только на ветке `main` и не
разворачивает сервис ни в каком окружении (ни staging, ни prod), поэтому
даже с этим шагом workflow не дотягивает до **CD** (continuous delivery/
deployment): нет ни автоматического деплоя, ни отдельного approve-шага для
выката, ни обновления работающего окружения по этому пушу.
