import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env_robust_23")

# ============================
# КЛИЕНТ
# ============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================
# ФУНКЦИЯ: кодируем картинку в base64
# ============================
def encode_image(image_path: str) -> str:
    """Читаем файл и кодируем в base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    allowed_formats = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in allowed_formats:
        raise ValueError(f"Неподдерживаемый формат: {ext}. Используйте: {allowed_formats}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================
# ФУНКЦИЯ: отправляем в Vision API
# ============================
def analyze_image(image_path: str, question: str = "Опиши что на изображении") -> str:
    """Отправляем изображение в OpenAI Vision API и получаем ответ."""
    print(f"  Загружаем: {image_path}")

    # Кодируем картинку
    image_data = encode_image(image_path)

    # Определяем тип файла
    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else f"image/{ext[1:]}"

    # Отправляем в API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        max_tokens=500,
    )

    return response.choices[0].message.content


# ============================
# ДЕМОНСТРАЦИЯ: 3 изображения разного типа
# ============================
def main():
    print("=" * 55)
    print("  🖼️  Vision API — Анализ изображений")
    print("=" * 55)

    # Три демо-сценария для интернет-магазина
    demos = [
        {
            "type": "📦 Фото товара",
            "path": "demo_product.jpg",
            "question": "Ты консультант интернет-магазина. Опиши этот товар: что это, в каком состоянии, есть ли видимые дефекты?"
        },
        {
            "type": "🖥️ Скриншот",
            "path": "demo_screenshot.png",
            "question": "Опиши что изображено на этом скриншоте. Какая это программа или сайт?"
        },
        {
            "type": "📊 График",
            "path": "demo_chart.png",
            "question": "Проанализируй этот график. Какие данные он показывает и какие выводы можно сделать?"
        },
    ]

    for demo in demos:
        print(f"\n{demo['type']}: {demo['path']}")
        print(f"Вопрос: {demo['question'][:60]}...")

        try:
            answer = analyze_image(demo["path"], demo["question"])
            print(f"Ответ: {answer[:300]}...")
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
        except ValueError as e:
            print(f"  ❌ {e}")
        except Exception as e:
            print(f"  ❌ Ошибка API: {e}")

    # Интерактивный режим
    print("\n" + "=" * 55)
    print("  Интерактивный режим — анализируй свои картинки")
    print("  Введите 'quit' для выхода")
    print("=" * 55)

    while True:
        path = input("\nПуть к изображению: ").strip()
        if path.lower() == "quit":
            print("До свидания!")
            break

        question = input("Вопрос (Enter = описать изображение): ").strip()
        if not question:
            question = "Опиши подробно что на изображении"

        try:
            print("\nАнализируем...")
            answer = analyze_image(path, question)
            print(f"\n🤖 Ответ:\n{answer}")
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
        except ValueError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()