from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    name: str
    input_price_per_1m: float
    output_price_per_1m: float
    context_window: int


AVAILABLE_MODELS = [
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        context_window=128000,
    ),
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        input_price_per_1m=5.00,
        output_price_per_1m=15.00,
        context_window=128000,
    ),
    ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        input_price_per_1m=10.00,
        output_price_per_1m=30.00,
        context_window=128000,
    ),
]
