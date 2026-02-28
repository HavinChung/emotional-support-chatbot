import requests

def call_llm(system_prompt, user_prompt, api_key, model=None):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    )
    result = response.json()
    if 'choices' not in result:
        raise ValueError(f"API Error: {result}")
    return result['choices'][0]['message']['content']