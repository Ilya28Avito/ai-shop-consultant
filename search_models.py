from huggingface_hub import HfApi

api = HfApi()

models = list(api.list_models(
    filter="text-classification",
    sort="downloads",
    limit=20
))

print("Топ модели для классификации тональности:\n")
for i, model in enumerate(models, 1):
    print(f"{i}. {model.id}")
    print(f"   Загрузок: {model.downloads}")
    print(f"   Лайков: {model.likes}")
    print()