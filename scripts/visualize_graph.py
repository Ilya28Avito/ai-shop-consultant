"""Блок 6.3 — сохраняет mermaid-схемы обоих графов из agent_graph.py.

Запуск: python scripts/visualize_graph.py
Результат: docs/agent-graph-custom.mmd, docs/agent-graph-prebuilt.mmd
(и docs/agent-graph.png, если в окружении есть playwright/pyppeteer —
опционально, отсутствие не считается ошибкой).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # python scripts/visualize_graph.py кладёт в sys.path только scripts/, а
    # не корень проекта — без этого `from app...` не найдётся.
    sys.path.insert(0, str(ROOT))

from app.services.agent_graph import custom_graph, prebuilt_graph

DOCS_DIR = ROOT / "docs"


def save_mermaid(graph, out_path: Path) -> None:
    mermaid_source = graph.get_graph().draw_mermaid()
    out_path.write_text(mermaid_source, encoding="utf-8")
    print(f"Сохранено: {out_path}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    save_mermaid(custom_graph, DOCS_DIR / "agent-graph-custom.mmd")
    save_mermaid(prebuilt_graph, DOCS_DIR / "agent-graph-prebuilt.mmd")

    # PNG — опционально, требует playwright/pyppeteer в окружении.
    try:
        png_bytes = custom_graph.get_graph().draw_mermaid_png()
        png_path = DOCS_DIR / "agent-graph.png"
        png_path.write_bytes(png_bytes)
        print(f"Сохранено: {png_path}")
    except Exception as exc:
        print(f"PNG не сохранён (не критично, остаются только .mmd): {exc}")


if __name__ == "__main__":
    main()
