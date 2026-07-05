import structlog
import logging


def setup_logging(level: str = "INFO") -> None:
    """Настройка structlog с JSON-выводом."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
