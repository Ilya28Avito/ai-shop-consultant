from openai import OpenAI

client = OpenAI(
    api_key="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"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Привет"}],
)

print(response.choices[0].message.content)