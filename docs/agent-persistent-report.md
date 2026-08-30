# Отчёт: agent_persistent — Блок 6.4 (персистентность, HIL, стриминг, time-travel)

`app/services/agent_persistent.py` — отдельный граф поверх тех же 3 tools, что
и в `agent_graph.py` (ДЗ 6.3), который сам `agent_graph.py` НЕ изменяет и
остаётся эталоном "голого" ReAct-графа без персистентности. Здесь добавлены:
переключаемый checkpointer (`memory`/`sqlite`/`postgres`), human-in-the-loop
через `interrupt()` + `Command(resume=...)` на одном "опасном" инструменте,
SSE-стриминг (`POST /agent/stream`) и time-travel демонстрация. Все выводы
ниже — реальные логи из тестирования этого ДЗ (curl, psql, pytest), не
синтетика.

## 1. Backend checkpointer'а: выбор по режиму

Переключатель — переменная окружения `AGENT_CHECKPOINTER` (`agent_lifespan()`
в `agent_persistent.py`), три значения:

- **`memory`** (`InMemorySaver`) — состояние живёт только в процессе, теряется
  при рестарте. Используется в собственных unit-тестах инструмента (не в
  `tests/test_agent_persistent.py` — там используется `sqlite(":memory:")`,
  см. ниже про причину этого выбора), и как самый дешёвый вариант для
  разовой ручной проверки логики графа без каких-либо файлов на диске.
- **`sqlite`** (`AsyncSqliteSaver`, дефолт) — один файл
  `var/agent_checkpoints.sqlite`, переживает рестарт процесса, не требует
  поднятого Postgres. Основной режим для локальной разработки — именно на
  нём отлаживался и подтверждён весь HIL-сценарий (см. разделы 3-4).
- **`postgres`** (`AsyncPostgresSaver`) — прод-режим. Требует `POSTGRES_URI`
  в окружении (обязателен, иначе `KeyError` — намеренно, чтобы не запускать
  прод-режим на молчаливо пустой строке подключения).

Во всех трёх случаях `checkpointer.setup()` вызывается **ровно один раз** —
внутри `agent_lifespan()`, при входе в `async with` в FastAPI `lifespan()`
(`app/main.py`), а не на каждый запрос к `/agent/stream`. Проверено логом
старта сервера:

```
{"backend": "sqlite", "event": "agent_persistent_ready", ...}
INFO:     Application startup complete.
```

— строка печатается один раз при старте процесса, не при каждом запросе.

## 2. Postgres: конфигурация и подтверждение

Важная деталь, зафиксированная честно, а не по умолчанию из формулировки
задания: Postgres в этом проекте раньше нигде не был настроен (ни в
`compose.yaml`, ни в `.env.example`, ни в `app/core/config.py`) — задание
предполагало переиспользование "того же" Postgres из блока "М3Б5", которого
в репозитории не оказалось. Поднят с нуля в этом ДЗ.

`compose.yaml`, сервис `postgres`:

```yaml
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: ai_shop
    POSTGRES_PASSWORD: ai_shop
    POSTGRES_DB: ai_shop
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ai_shop -d ai_shop"]
  restart: unless-stopped
```

`app`-сервис получает `POSTGRES_URI` и `AGENT_CHECKPOINTER: postgres` через
`environment`, плюс `depends_on: postgres: condition: service_healthy`.
Локально вне Docker (CLI на хосте) — та же строка подключения с `localhost`:

```
POSTGRES_URI=postgresql://ai_shop:ai_shop@localhost:5432/ai_shop
```

Проверка `setup()` вживую — сначала пусто, затем прогон CLI с явным
`AGENT_CHECKPOINTER=postgres`, затем снова `\dt`:

```
D:\ai-shop>docker compose exec postgres psql -U ai_shop -d ai_shop -c "\dt"
Did not find any relations.

D:\ai-shop>python -m app.services.agent_persistent "Который час в Москве?" postgres-test
...
В Москве сейчас 12:35.

D:\ai-shop>docker compose exec postgres psql -U ai_shop -d ai_shop -c "\dt"
                List of relations
 Schema |         Name          | Type  |  Owner
--------+-----------------------+-------+---------
 public | checkpoint_blobs      | table | ai_shop
 public | checkpoint_migrations | table | ai_shop
 public | checkpoint_writes     | table | ai_shop
 public | checkpoints           | table | ai_shop
(4 rows)
```

Четыре таблицы созданы ровно тем самым единственным вызовом `setup()` внутри
`agent_lifespan()` — backend `postgres` подтверждён рабочим сквозным
сценарием, не только чтением кода. (О баге, из-за которого первая попытка
этого прогона упала — см. раздел 8, пункт про `ProactorEventLoop`.)

## 3. Опасный инструмент и идемпотентность HIL

Выбран `send_telegram_message(chat_id, text)` — необратимая коммуникация с
реальным человеком (в отличие от `search_knowledge_base` и `get_current_time`,
у которых нет побочных эффектов). Реализован через **два отдельных узла**,
а не `interrupt_before`/`interrupt_after` (устаревший паттерн):

- **`prepare_send_telegram`** — идемпотентный: только читает уже сделанный
  моделью `tool_call` и собирает `pending_action` (chat_id, text, tool_call_id)
  в state. Никакого side-effect. Безопасно переиграть сколько угодно раз.
- **`confirm_and_execute_send_telegram`** — до `interrupt()` только читает
  `pending_action` (тоже безопасно для replay); сам `print(f"[TELEGRAM →
  ...]")` (в реальной интеграции — вызов Telegram Bot API) стоит **строго
  после** `interrupt()`.

Это разделение — не формальность. `interrupt()` работает через replay: при
каждом `Command(resume=...)` LangGraph переигрывает узел `confirm_and_execute_
send_telegram` заново с начала, и `interrupt()` на этот раз возвращает
значение resume вместо повторной остановки. Всё, что в узле находится ДО
`interrupt()`, физически выполняется на каждом таком replay. Если бы сам
`print`/реальная отправка стояли до `interrupt()`, при каждом дополнительном
resume (например, повторном по сетевой ошибке клиента) сообщение отправлялось
бы клиенту заново — задание прямо называет это "миной замедленного действия".
Вынос side-effect строго после `interrupt()` устраняет этот риск: тест
`test_resume_true_sends_message` проверяет, что print-стаб вызывается ровно
один раз за один `pending_action`.

Известное упрощение `execute_tool` (безопасные tool'ы): если модель в одном
шаге запросит параллельно и безопасный, и опасный tool (`parallel tool_calls`),
исполнится только безопасный — опасный останется без ответа. На практике не
наблюдалось благодаря ограничению "не более одного инструмента за шаг" в
`SYSTEM_PROMPT`, но это не гарантия самого API. См. раздел 8.

## 4. Interrupt: payload и лог после resume

Реальный прогон через SSE (`thread_id=sse-demo-4`, задача "отправь клиенту в
чат 555 сообщение: заказ готов к выдаче"):

```
data: {"type": "update", "node": "call_model"}
data: {"type": "update", "node": "prepare_send_telegram"}
data: {"type": "interrupt", "payload": {"type": "approve_send_telegram_message",
       "preview": "Отправить в Telegram (chat_id=555): 'заказ готов к выдаче'"}}
data: {"type": "done"}
```

Resume (`{"thread_id": "sse-demo-4", "input": {"resume": true}}`):

```
data: {"type": "token", "node": "confirm_and_execute_send_telegram",
       "content": "Сообщение отправлено в 555"}
data: {"type": "update", "node": "confirm_and_execute_send_telegram"}
data: {"type": "token", "node": "call_model", "content": "Сообщение отправлено в чат 555."}
data: {"type": "update", "node": "call_model"}
data: {"type": "update", "node": "force_finish"}
data: {"type": "done"}
```

И, что важнее самого SSE-потока — реальный side-effect в логе сервера,
выполнившийся строго один раз, строго после resume:

```
[TELEGRAM → 555] заказ готов к выдаче
```

Замечание по формату SSE: чанк от `confirm_and_execute_send_telegram` пришёл
одним куском, а не по токенам, как у `call_model`. Это ожидаемо: `stream_mode
="messages"` стримит токены только у реальных LLM-вызовов; сообщение, которое
узел просто возвращает напрямую (без обращения к модели), тоже проходит по
каналу `"messages"`, но одним целым чанком.

## 5. Time-travel: история чекпоинтов и два исхода

`scripts/time_travel_demo.py` — офлайн в смысле персистентности
(`AsyncSqliteSaver(":memory:")`, ни var-файл, ни Postgres не трогает).
Ключевой методологический момент, вынесенный в докстринг модуля: значение
`Command(resume=...)` фиксируется в checkpointer'е как pending write для
конкретного `(thread_id, checkpoint_id)` и детерминировано — повторный resume
ТОГО ЖЕ checkpoint'а с другим значением не изменит уже зафиксированный исход.
Поэтому единственный честный способ показать оба исхода (подтверждение и
отказ) на идентичной задаче — два РАЗНЫХ `thread_id`, не повторный resume
одного треда. Ранняя попытка воспроизвести это через `sse-demo-2`/`sse-demo-3`
на одном и том же треде (см. раздел 8) наглядно это подтвердила на практике,
хоть и по другой причине (отсутствие системного промпта), чем ожидалось для
демонстрации time-travel — но сам принцип "не резюмить один и тот же
checkpoint дважды с разным ожиданием" от этого не менее верен.

Реальный вывод скрипта (оба треда, полностью):

```
=== thread_id='time-travel-approve' (решение: approve) ===
Payload interrupt(): {'type': 'approve_send_telegram_message',
  'preview': "Отправить в Telegram (chat_id=42): 'ваш заказ передан курьеру'"}

История чекпоинтов (4 шт., от новых к старым):
  1f1a455c-84c6-639e-8002-61deef9cf8fb   next=('confirm_and_execute_send_telegram',)
  1f1a455c-84c4-678c-8001-367cca451aba   next=('prepare_send_telegram',)
  1f1a455c-7787-61b3-8000-2e9160c90bf7   next=('call_model',)
  1f1a455c-7784-6a5d-bfff-fd33321ac0b2   next=('__start__',)

Чтение прошлого состояния (checkpoint_id=1f1a455c-7784-6a5d-bfff-fd33321ac0b2):
  iteration_count = None
  messages = 0 сообщений

[TELEGRAM → 42] ваш заказ передан курьеру

После resume(decision=True): sent=True
Финальный ответ модели: Сообщение отправлено в чат 42.

=== thread_id='time-travel-reject' (решение: reject) ===
Payload interrupt(): {'type': 'approve_send_telegram_message',
  'preview': "Отправить в Telegram (chat_id=42): 'ваш заказ передан курьеру'"}

История чекпоинтов (4 шт., от новых к старым):
  1f1a455c-97cb-6564-8002-a357047adf92   next=('confirm_and_execute_send_telegram',)
  1f1a455c-97c9-622d-8001-3215b7ec9800   next=('prepare_send_telegram',)
  1f1a455c-8eaa-6267-8000-66cfa36c8800   next=('call_model',)
  1f1a455c-8ea7-6601-bfff-c9a442b0ecdc   next=('__start__',)

Чтение прошлого состояния (checkpoint_id=1f1a455c-8ea7-6601-bfff-c9a442b0ecdc):
  iteration_count = None
  messages = 0 сообщений

После resume(decision=False): sent=False
Финальный ответ модели: Не удалось отправить сообщение: подтверждение на отправку не было получено.
```

Одинаковый входной запрос, идентичная структура истории (4 чекпоинта:
`__start__` → `call_model` → `prepare_send_telegram` →
`confirm_and_execute_send_telegram`), но два по-настоящему разных исхода —
именно потому, что это два разных `thread_id`. Самый старый чекпоинт
(`__start__`) читается отдельно от текущего состояния треда — это и есть
"машина времени": `aget_state({"configurable": {"thread_id": ..., "checkpoint_
id": <старый>}})` возвращает снимок ДО того, как что-либо выполнилось
(`iteration_count = None`, `messages = 0` — граф ещё не успел дописать даже
первый ответ модели).

## 6. Стриминг: выбор режима

`graph.astream(input, config, stream_mode=["updates", "messages"])` вместо
`astream_events(version="v2")`. `astream_events` даёт более гранулярный
event-tree (`on_chat_model_start`/`on_chat_model_stream`/`on_tool_start` и
т.п.) — удобно для сложного UI с единым потоком событий, но для двух
категорий, которые реально нужны здесь (прогресс по узлам графа + токены
модели по мере генерации, включая момент `__interrupt__`), это избыточная
детализация и заметно больший объём трафика на SSE-соединении.
`stream_mode=["updates", "messages"]` покрывает обе категории напрямую:
`"updates"` даёт факт завершения узла (и специальный ключ `"__interrupt__"`
с payload'ом), `"messages"` — токены с привязкой к узлу через
`metadata["langgraph_node"]`. Пример реального вывода — раздел 4 выше.

## 7. Политика доступа: user_role

`config["configurable"]` теперь содержит `user_role` (`POST /agent/stream`
принимает необязательное поле `user_role`, дефолт `"write_with_approve"`).
Сама возможность передать роль по цепочке вызовов заведена, но
`confirm_and_execute_send_telegram` её пока не читает и `interrupt()`
вызывает всегда, независимо от роли — это сознательное решение: пропуск
подтверждения для роли `full` — бизнес-решение о доверии конкретным ролям
(например, внутренний оператор поддержки против анонимного пользователя
бота), которое нужно принимать отдельно от инженерной части этого ДЗ, а не
зашивать в код по умолчанию.

## 8. Что осталось хрупким / TODO

**Баг 1 (найден и исправлен) — `SYSTEM_PROMPT` не подставлялся.** Константа
была объявлена с корректной инструкцией про HIL ("просто вызови инструмент
как обычно, подтверждение граф запросит сам"), но нигде не попадала в
`messages` — ни в CLI, ни в `/agent/stream`. Граф работал вообще без
системного промпта. Обнаружено эмпирически: два подряд SSE-теста на
`send_telegram_message` (`thread_id=sse-demo-2`, `sse-demo-3`) оба привели к
тому, что модель ПРОСИЛА подтверждение текстом вместо вызова tool'а — хотя
более ранний CLI-тест похожую задачу отработал верно (по собственному
умолчанию модели, без какой-либо системной инструкции — чистая случайность).
Исправлено добавлением `build_initial_state()` — единственной точки входа
для сборки state нового треда, подставляющей `SYSTEM_PROMPT` первым
сообщением. Подтверждено повторным SSE-тестом на свежем `thread_id=sse-demo-
4` — `interrupt()` сработал стабильно (раздел 4). Побочный урок: `sse-demo-3`
при ПОВТОРНОМ запуске после фикса всё равно не сработал — потому что это
старый `thread_id`, чья история в checkpointer'е уже была "заражена"
отсутствием системного промпта с первого сообщения, а `build_initial_state()`
применяется только при старте НОВОГО треда, не переписывает существующую
историю. На практике это означает: смена системного промпта задним числом не
долечивает уже существующие треды — для настоящей эволюции промпта
понадобится отдельная стратегия миграции/версионирования истории, которой в
этом ДЗ нет.

**Баг 2 (найден и исправлен) — `AsyncPostgresSaver` + Windows.** Первая
попытка прогнать `AGENT_CHECKPOINTER=postgres` на Windows упала:

```
psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in
async mode. Please use a compatible event loop...
```

Известная, задокументированная в самом psycopg несовместимость: асинхронный
режим psycopg не работает с `ProactorEventLoop` — дефолтным event loop'ом
asyncio на Windows. Исправлено переключением политики event loop'а на
`WindowsSelectorEventLoopPolicy()` при импорте `agent_persistent.py`, если
`sys.platform == "win32"`. Подтверждено повторным прогоном (раздел 2) — все 4
таблицы созданы. Caveat на будущее: `asyncio.set_event_loop_policy` и сам
`WindowsSelectorEventLoopPolicy` помечены deprecated начиная с текущей версии
Python (предупреждение подтверждено в реальном выводе — "slated for removal
in Python 3.16"), так что при обновлении Python эту строчку придётся
переписать на актуальный API переключения event loop'а. В Docker (Linux, тот
же `compose.yaml`) эта ветка кода не выполняется вовсе — там несовместимости
нет по построению, так что "прод"-путь через `docker compose up` данного бага
не касается в принципе.

**Известное упрощение (не баг, задокументированное решение)** —
`execute_tool` не обрабатывает параллельные `tool_calls`, где один вызов
безопасный, а другой опасный: исполнится только безопасный, второй останется
без ответа. Не воспроизводилось на практике благодаря ограничению "один
инструмент за шаг" в `SYSTEM_PROMPT`, но само по себе ограничение — это
дисциплина промпта, а не гарантия API.

**Наблюдение по стримингу (не баг)** — сообщения, которые узел возвращает
напрямую (без обращения к LLM), в `stream_mode="messages"` приходят одним
целым чанком, а не по токенам, как настоящая генерация `call_model`. Клиенту
SSE-потока, если он рассчитывает на равномерный token-by-token UX, стоит
учитывать эту разницу отдельно (раздел 4, раздел 6).
