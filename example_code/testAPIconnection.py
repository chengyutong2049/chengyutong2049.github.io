from openai import OpenAI

client = OpenAI(api_key="...")
response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        # prompt='Be short and precise"',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0,
        max_tokens=100
    )

print(response['choices'][0]['message']['content'])