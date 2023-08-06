import requests
import json
import os
import argparse
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
    'HTTP-Referer': 'https://www.google.com/'
}

# Initial system message
messages = [{'role': 'system', 'content': 'You are a helpful assistant designed to serve the user.'}]

def combine_text(record):
    instruction = record['instruction']
    response = record['output']
    combined_text = f"You are a helpful coding assistant, below is an instruction that you have to complete by analysing user request and planning how to complete it. Then respond as required step-by-step.\n\n### Instruction:\n\n{instruction}\n\n### Response:\n\n{response}"
    return combined_text

parser = argparse.ArgumentParser()
parser.add_argument('infile', help='Input JSONL file')
parser.add_argument('outfile', help='Output JSONL file')
parser.add_argument('--lines', type=int, help='Number of lines to read from the input file')
args = parser.parse_args()

with open(args.infile, 'r') as infile, open(args.outfile, 'w') as outfile:
    for i, line in enumerate(infile):
        # Stop reading after the specified number of lines
        if args.lines is not None and i >= args.lines:
            break

        record = json.loads(line)
        
        # Submit the instruction as the user message
        user_message = record['instruction']
        messages.append({'role': 'user', 'content': user_message})
        
        data = {
            'model': MODEL,
            'messages': messages
        }
        
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, data=json.dumps(data))
        
        try:
            response_json = response.json()
            # Extract the assistant's message from the response
            assistant_message = response_json['choices'][0]['message']['content']
            # Add the assistant's message to the conversation
            messages.append({'role': 'assistant', 'content': assistant_message})

            # Update the record with the assistant's response
            record['output'] = assistant_message
            # Add the combined text field
            record['text'] = combine_text(record)
            # Write the updated record to the output file
            outfile.write(json.dumps(record) + '\n')
            
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            print("Status code:", response.status_code)
            print("Response text:", response.text)
