import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
YOUR_SITE_URL = 'rimer.world'  # replace 'your-site-url' with your actual site URL
MODEL_ACRONYM = 'GPT4'  # replace with your chosen model acronym

# Map acronyms to full model names
MODEL_MAP = {
    'GPT3TURBO': 'openai/gpt-3.5-turbo',
    'GPT3TURBO16K': 'openai/gpt-3.5-turbo-16k',
    'GPT4': 'openai/gpt-4',
    'GPT432K': 'openai/gpt-4-32k',
    'SHAPE': 'openai/shap-e',
    'BISON': 'google/palm-2-chat-bison',
    'CODEBISON': 'google/palm-2-codechat-bison',
    'CLAUDEINSTANT': 'anthropic/claude-instant-v1',
    'CLAUDEINSTANT100K': 'anthropic/claude-instant-v1-100k',
    'CLAUDEV1': 'anthropic/claude-v1',
    'CLAUDEV1100K': 'anthropic/claude-v1-100k',
    'CLAUDEINSTANT1': 'anthropic/claude-instant-1.0',
    'CLAUDEV12': 'anthropic/claude-1.2',
}

MODEL = MODEL_MAP[MODEL_ACRONYM]

headers = {
    'Authorization': 'Bearer ' + OPENROUTER_API_KEY,
    'HTTP-Referer': YOUR_SITE_URL  # To identify your app
}

# Initial system message
messages = [{'role': 'system', 'content': 'You are a helpful assistant designed to serve the user.'}]

while True:
    user_message = input("You: ")
    
    # Check if the user wants to quit
    if user_message.lower() == 'quit':
        break

    # Add the user's message to the conversation
    messages.append({'role': 'user', 'content': user_message})

    data = {
        'model': MODEL,
        'messages': messages
    }

    response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, data=json.dumps(data))

    try:
        response_json = response.json()

        # Extract the assistant's message from the response and print it
        assistant_message = response_json['choices'][0]['message']['content']
        print("Assistant: " + assistant_message)

        # Add the assistant's message to the conversation
        messages.append({'role': 'assistant', 'content': assistant_message})
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        print("Status code:", response.status_code)
        print("Response text:", response.text)