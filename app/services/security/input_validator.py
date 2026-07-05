import re
from dataclasses import dataclass
from typing import Final

INJECTION_PATTERNS: Final[list] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(the\s+)?(system|previous|above)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(a|an|the|dan|do anything now)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all|previous)\b", re.IGNORECASE),
    re.compile(r"\b(jailbroken|developer mode|godmode)\b", re.IGNORECASE),
    re.compile(r"base64", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
]

MAX_INPUT_CHARS: Final[int] = 4000
NON_PRINTABLE_RATIO_LIMIT: Final[float] = 0.10


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    rule: str | None = None


def validate_input(text: str) -> ValidationResult:
    """Валидация входящего текста на инъекции."""
    if len(text) > MAX_INPUT_CHARS:
        return ValidationResult(False, "input too long", rule="length")

    non_printable = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
    if non_printable / max(len(text), 1) > NON_PRINTABLE_RATIO_LIMIT:
        return ValidationResult(False, "high non-printable ratio", rule="encoding")

    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return ValidationResult(False, f"matched pattern: {pat.pattern}", rule="injection")

    return ValidationResult(True)
