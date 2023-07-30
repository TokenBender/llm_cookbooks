import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
YOUR_SITE_URL = os.getenv('YOUR_SITE_URL')
MODEL_ACRONYM = 'BISON'

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
    'HTTP-Referer': YOUR_SITE_URL
}

# Specify the input columns
input_columns = ['column1', 'column2']

# Define the processing function
def process_row(row):
    # This function takes a row of data and returns a dictionary of {column_name: new_value} pairs.
    # The new values will be used to update the row.
    # This is where you would add your own processing logic.

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant designed to take in rows of a dataset from a user and .'},
        {'role': 'user', 'content': row['column1']}
    ]

    data = {
        'model': MODEL,
        'messages': messages
    }

    response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, data=json.dumps(data))

    try:
        response_json = response.json()
        assistant_message = response_json['choices'][0]['message']['content']

        # Return a dictionary of {column_name: new_value} pairs
        return {'column1': assistant_message}

    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        print("Status code:", response.status_code)
        print("Response text:", response.text)
        return {}

# Use chunksize to load and process data in batches
chunksize = 4
chunks = []
for chunk in pd.read_csv('data.csv', chunksize=chunksize):
    for idx, row in chunk.iterrows():
        # Use the processing function to update the row
        chunk.loc[idx, input_columns] = process_row(row)

    chunks.append(chunk)

# Concatenate all chunks and save to a new JSON file
df = pd.concat(chunks)
df.to_json('data_augmented.json', orient='records', lines=True)
