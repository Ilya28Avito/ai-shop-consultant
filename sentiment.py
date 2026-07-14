from transformers import pipeline
import time

# Тестовые отзывы покупателей интернет-магазина
reviews = [
    "Отличный товар, очень доволен покупкой!",
    "Ужасное качество, деньги на ветер",
    "Нормально, ничего особенного",
    "Быстрая доставка, товар соответствует описанию",
    "Брак! Сломалось на второй день",
    "Супер! Рекомендую всем друзьям",
    "Долго ждал, но в целом доволен",
    "Не буду больше заказывать здесь",
    "Хорошее соотношение цены и качества",
    "Полное разочарование, не то что на фото",
]

MODELS = [
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "blanchefort/rubert-base-cased-sentiment",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
]

for model_name in MODELS:
    print(f"\n{'='*50}")
    print(f"Модель: {model_name}")
    print('='*50)
    
    try:
        classifier = pipeline("text-classification", model=model_name)
        
        start = time.perf_counter()
        results = classifier(reviews)
        elapsed = time.perf_counter() - start
        
        for review, result in zip(reviews, results):
            print(f"  [{result['label']}] {review[:40]}...")
        
        print(f"\n  Время inference: {elapsed:.2f}s")
        print(f"  Среднее на отзыв: {elapsed/len(reviews):.3f}s")
        
    except Exception as e:
        print(f"  Ошибка: {e}")