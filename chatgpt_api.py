import openai

def generate_prompt(question):
    return """{} 
    Question:{} 
    Answer:""".format("SYSTEM_PROMPT",
        question
    )

def generate_response_chatgpt(api_key, question):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": generate_prompt(question)}
        ]
        )
    return response['choices'][0]['message']['content']