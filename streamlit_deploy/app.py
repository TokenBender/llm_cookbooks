import streamlit as st
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import base64
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

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

headers = {
    'Authorization': 'Bearer ' + OPENROUTER_API_KEY,
    'HTTP-Referer': 'https://www.google.com/'
}

# Initial system message
messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

def combine_text(instruction, response):
    combined_text = f"You are a helpful coding assistant.\n\n### Instruction:\n\n{instruction}\n\n### Response:\n\n{response}"
    return combined_text

st.title('JSONL File Processing')

with st.form(key='my_form'):
    model_choice = st.selectbox('Select model', list(MODEL_MAP.keys()))
    MODEL = MODEL_MAP[model_choice]

    uploaded_file = st.file_uploader("Upload JSONL file", type=['jsonl'])

    num_lines = st.number_input('Number of lines to process', min_value=1, step=1, value=5)
    start_line = st.number_input('Start processing from line', min_value=0, step=1, value=0)

    output_file = st.text_input('Output file name', 'output.jsonl')

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if uploaded_file is not None:
        data_list = []
        
        for i, line in enumerate(uploaded_file):
            if i < start_line:
                continue
            if len(data_list) >= num_lines:
                break
            data_list.append(json.loads(line))

        df = pd.DataFrame(data_list)
        st.write(df)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        for i, record in df.iterrows():
            # Update the progress bar
            progress_bar.progress((i + 1) / len(df))
            
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
                assistant_message = response_json['choices'][0]['message']['content']
                messages.append({'role': 'assistant', 'content': assistant_message})

                # Update the record with the assistant's response
                df.at[i, 'output'] = assistant_message
                # Add the combined text field
                df.at[i, 'text'] = combine_text(record['instruction'], assistant_message)

            except json.JSONDecodeError:
                st.error("Failed to parse JSON response.")
        
        st.write(df)

        # Convert DataFrame to JSONLines string
        jsonl = "\n".join(df.apply(lambda x: x.to_json(), axis=1))

        # Encode JSONLines string to bytes, then to Base64
        b64 = base64.b64encode(jsonl.encode()).decode()

        href = f'<a href="data:file/json;base64,{b64}" download="{output_file}">Download {output_file}</a>'
        st.markdown(href, unsafe_allow_html=True)