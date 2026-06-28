class LLMError(Exception):
    """Базовое исключение для LLM ошибок."""
    def __init__(self, message: str, code: str = "llm_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Превышен лимит запросов."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, code="llm_rate_limit")


class LLMTimeoutError(LLMError):
    """Таймаут запроса."""
    def __init__(self, message: str = "Request timeout"):
        super().__init__(message, code="llm_timeout")


class LLMAuthError(LLMError):
    """Ошибка аутентификации."""
    def __init__(self, message: str = "Authentication error"):
        super().__init__(message, code="llm_auth")
