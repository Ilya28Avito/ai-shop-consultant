from functools import lru_cache
from pathlib import Path
from jinja2 import Template

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=8)
def render_system_prompt(version: str = "v1", **context) -> str:
    """Загружаем system prompt из файла и рендерим через Jinja2."""
    text = (PROMPTS_DIR / f"system_{version}.j2").read_text(encoding="utf-8")
    return Template(text).render(**context)
