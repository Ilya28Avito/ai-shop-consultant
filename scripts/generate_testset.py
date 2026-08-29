"""
ДЗ 5.6, шаг 2 — генерация "сырого" golden dataset через RAGAS TestsetGenerator.

Что делает: берёт ВЕСЬ корпус data/ (рекурсивно, все *.md — те же файлы,
что индексирует scripts/ingest.py через Path(data_dir).rglob("*.md"), включая
подпапки вроде rag-block-03), прогоняет через RAGAS TestsetGenerator и
сохраняет результат as-is в tests/eval/golden_dataset_raw.csv.

Документы читаем вручную (Path.read_text), НЕ через langchain DirectoryLoader —
у него дефолтный loader_cls тянет пакет `unstructured` (+ NLTK-данные) для .md,
это лишняя тяжёлая зависимость, которая нам не нужна.

ВАЖНО: это только СЫРОЙ автогенерированный результат. Дальше — обязательная
ручная вычитка (шаг 2Б, руками, не скриптом): открыть CSV, выкинуть дубли,
слишком общие вопросы ("Что такое X?"), нелепые, доразметить reference там,
где модель промахнулась мимо контента data/. Итог сохранить в
tests/eval/golden_dataset.json с полями user_input, reference,
reference_contexts, минимум 30 строк. Без вычитки этот шаг не засчитывается
(см. критерии самопроверки в задании) — мусорный golden обнулит все метрики
на шаге 3.

Запуск (сначала маленький прогон, чтобы проверить, что всё работает):
    python scripts/generate_testset.py --size 5

Потом полный:
    python scripts/generate_testset.py --size 35
"""
import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
# override=True: явно приоритизируем .env_robust_23 над системными переменными окружения
# Windows — иначе, если где-то уже прописан свой OPENAI_API_KEY, load_dotenv молча
# оставит его как есть и в OpenAI улетит не тот ключ.
load_dotenv(".env_robust_23", override=True)

import openai
from langchain_core.documents import Document
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator

DATA_DIR = "data"
OUT_PATH = Path("tests/eval/golden_dataset_raw.csv")

# Тот же судья, что и в run_eval.py (gpt-4o-mini) — единообразие и контроль коста.
# Генерация тестсета делает МНОГО вызовов LLM (построение knowledge graph +
# синтез вопросов), это не бесплатно и не мгновенно — на --size 35 может уйти
# несколько минут и заметное число токенов. Поэтому сначала тестируем на --size 5.
GENERATOR_MODEL = "gpt-4o-mini"


# CommonMark-экранированные спецсимволы (\# \* \_ и т.п.) — часть корпуса
# (catalog/, guides/, policies/, support/) хранит markdown-заголовки экранированными
# (буквально "\# Заголовок" вместо "# Заголовок"). Для человека/Qdrant это не проблема
# (текст читается нормально), но ragas.testset.HeadlineSplitter ищет строки, начинающиеся
# ровно с "#", и с обратным слэшем их не видит -> падает с
# "'headlines' property not found in this node". Снимаем экранирование только для
# генерации testset'а, сами файлы в data/ не трогаем.
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>])")


def _unescape_markdown(text: str) -> str:
    return _MD_ESCAPE_RE.sub(r"\1", text)


# ragas.testset.transforms.default_transforms() сам решает, включать ли
# HeadlineSplitter, по эвристике: если >=25% документов длиннее 500 токенов —
# включает построение knowledge graph по заголовкам. У нас несколько гайдов
# (2-3.6 КБ кириллицы, это заметно больше 500 токенов) в эту категорию попадают,
# и после снятия экранирования часть узлов всё равно остаётся без 'headlines'
# (LLM-экстрактор не на всех текстах находит структуру) -> генерация падает.
# У generate_with_langchain_docs нет параметра, чтобы это отключить, поэтому
# дробим файлы на куски поменьше ЗАРАНЕЕ (по абзацам, с ориентиром на размер) —
# тогда ни один "документ" не пересечёт порог 500 токенов, и весь этот путь
# просто не включится.
_PARAGRAPH_CHUNK_TARGET_CHARS = 800


def _chunk_by_paragraphs(text: str, target_chars: int = _PARAGRAPH_CHUNK_TARGET_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current and current_len + len(para) > target_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_corpus(data_dir: str = DATA_DIR) -> list[Document]:
    """Загружает файлы из data/ (те же, что индексирует scripts/ingest.py),
    режет каждый на куски по абзацам (~800 символов) и возвращает как список
    langchain Document — по документу на кусок, а не на файл."""
    paths = sorted(Path(data_dir).rglob("*.md"))
    if not paths:
        raise SystemExit(
            f"В '{data_dir}' не нашлось .md файлов. Запускать скрипт нужно из корня "
            f"проекта (там, где лежит папка data/), например: python scripts/generate_testset.py"
        )
    docs = []
    for p in paths:
        text = _unescape_markdown(p.read_text(encoding="utf-8"))
        for i, chunk in enumerate(_chunk_by_paragraphs(text)):
            docs.append(Document(page_content=chunk, metadata={"source": str(p), "chunk": i}))
    return docs


def main(testset_size: int) -> None:
    docs = load_corpus()
    n_files = len({d.metadata["source"] for d in docs})
    print(f"Загружено файлов из {DATA_DIR}/: {n_files}, после нарезки по абзацам -> кусков: {len(docs)}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY не найден — проверь .env_robust_23")

    generator_llm = llm_factory(GENERATOR_MODEL, client=openai.OpenAI(api_key=api_key))
    generator_embeddings = OpenAIEmbeddings(client=openai.OpenAI(api_key=api_key))

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    print(f"\nГенерируем {testset_size} пар через RAGAS TestsetGenerator...")
    print("(идут вызовы LLM — построение knowledge graph + синтез вопросов, может занять пару минут)\n")

    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)

    df = dataset.to_pandas()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig — пишем с BOM-меткой в начале файла. Без неё Excel на русской локали
    # Windows при обычном двойном клике определяет кодировку "на глаз" и часто
    # ошибается на Windows-1251, превращая кириллицу в "РљР°С‚Р°Р»РѕРі..." (мохибейк).
    # BOM явно говорит Excel "это UTF-8" — он открывает файл кириллицей сразу, без
    # танцев с Data -> From Text/CSV. Для Python/pandas.read_csv BOM не проблема
    # в любом случае (см. utf-8-sig на чтении в csv_to_golden_json.py).
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Сохранено {len(df)} строк -> {OUT_PATH}")
    print(f"Столбцы: {list(df.columns)}")
    print(
        "\nСЛЕДУЮЩИЙ ШАГ — РУКАМИ, не скриптом:\n"
        "  1. Открой tests/eval/golden_dataset_raw.csv (Excel/LibreOffice или в редакторе).\n"
        "  2. Выкинь дубли, слишком общие вопросы ('Что такое X?'), нелепые/бессмысленные.\n"
        "  3. Проверь reference по каждой строке — поправь там, где модель промахнулась\n"
        "     мимо реального содержимого data/.\n"
        "  4. Сохрани итог как tests/eval/golden_dataset.json (список объектов с полями\n"
        "     user_input, reference, reference_contexts), минимум 30 записей после вычитки."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ДЗ 5.6, шаг 2 — генерация сырого golden dataset")
    parser.add_argument(
        "--size", type=int, default=35,
        help="Сколько пар генерировать (берём с запасом над минимумом 30 — после ручной вычитки часть уйдёт в брак)",
    )
    args = parser.parse_args()
    main(args.size)
