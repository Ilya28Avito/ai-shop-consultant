import gradio as gr
from transformers import pipeline

# Загружаем лучшую модель
classifier = pipeline(
    "text-classification",
    model="blanchefort/rubert-base-cased-sentiment"
)

def analyze_sentiment(text):
    if not text.strip():
        return "Введите текст отзыва"
    
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    
    if label == "POSITIVE":
        emoji = "😊"
        verdict = "Положительный"
    elif label == "NEGATIVE":
        emoji = "😞"
        verdict = "Отрицательный"
    else:
        emoji = "😐"
        verdict = "Нейтральный"
    
    return f"{emoji} {verdict} (уверенность: {score:.0%})"

# Создаём интерфейс
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Введите отзыв покупателя",
        placeholder="Например: Отличный товар, очень доволен покупкой!",
        lines=3
    ),
    outputs=gr.Textbox(label="Результат анализа"),
    title="🛍️ Анализ тональности отзывов",
    description="ИИ-консультант для интернет-магазина. Определяет тональность отзывов покупателей.",
    examples=[
        ["Отличный товар, очень доволен покупкой!"],
        ["Ужасное качество, деньги на ветер"],
        ["Нормально, ничего особенного"],
        ["Быстрая доставка, рекомендую!"],
        ["Брак! Сломалось на второй день"],
    ]
)

demo.launch(share=True)