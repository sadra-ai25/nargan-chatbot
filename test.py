from openai import OpenAI

client = OpenAI(
    api_key="sk-kFIVdGGKBeqL3Eb7nZ67CVBcNiuo5KEid6S8meSfNzcQPrsZ",
    base_url='https://api.gapgpt.app/v1'
)

response = client.chat.completions.create(
    model='gapgpt-qwen-3.5',
    messages=[
        {'role': 'user', 'content': 'سلام! یک تابع پایتون برای مرتب‌سازی لیست بنویس.'}
    ]
)

print(response.choices[0].message.content)
