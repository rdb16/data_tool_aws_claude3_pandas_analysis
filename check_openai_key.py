from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv('~/.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an helpful assistant."},
        {"role": "user", "content": "write a haiku about ai."}
    ]
)
print(completion.choices[0].message)
