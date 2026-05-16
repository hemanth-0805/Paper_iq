import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
print("API KEY loaded:", bool(api_key))

try:
    from openai import OpenAI
    print("OpenAI library loaded")
    client = OpenAI(api_key=api_key)
    print("Client created, making test request...")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=10,
    )
    print("Success:", resp.choices[0].message.content)
except Exception as e:
    print("Error occurred:")
    import traceback
    traceback.print_exc()
