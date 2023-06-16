import openai
from mySecrets import OpenAI_API_KEY

def chatGPT3_response(user_input):
    openai.api_key = OpenAI_API_KEY
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": user_input}]
        )
    return res["choices"][0]["message"]["content"]

