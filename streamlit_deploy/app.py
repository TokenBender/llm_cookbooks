import streamlit as st
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import base64
import os
import re

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Map acronyms to full model names
MODEL_MAP = {
    'GPT3TURBO': 'openai/gpt-3.5-turbo',
    'GPT3TURBO16K': 'openai/gpt-3.5-turbo-16k',
    'GPT4': 'openai/gpt-4',
    'GPT4-32K': 'openai/gpt-4-32k',
    'SHAPE': 'openai/shap-e',
    'BISON': 'google/palm-2-chat-bison',
    'CODEBISON': 'google/palm-2-codechat-bison',
    'CLAUDEINSTANT': 'anthropic/claude-instant-v1',
    'CLAUDEINSTANT100K': 'anthropic/claude-instant-v1-100k',
    'CLAUDEV1': 'anthropic/claude-v1',
    'CLAUDEV1100K': 'anthropic/claude-v1-100k',
    'CLAUDEINSTANT1': 'anthropic/claude-instant-1.0',
    'CLAUDEV12': 'anthropic/claude-1.2',
    'LLAMA2-13': 'meta-llama/llama-2-13b-chat',
    'LLAMA2-70B': 'meta-llama/llama-2-70b-chat',
}

# Set up headers for the API request
headers = {
    'Authorization': 'Bearer ' + OPENROUTER_API_KEY,
    'HTTP-Referer': 'https://www.google.com/'
}

# Function to combine instruction and response
def combine_text(instruction, response):
    combined_text = f"You are a helpful coding assistant.\\n\\n### Instruction:\\n\\n{instruction}\\n\\n### Response:\\n\\n{response}"
    return combined_text

# Function to check if a string contains Chinese characters
def contains_chinese(text):
    if re.search('[\\u4e00-\\u9fff]', text):
        return True
    return False

# Set up Streamlit interface
st.title('JSONL File Processing')

# User selects the processing mode
processing_mode = st.sidebar.selectbox("Choose the processing mode", ["With AI assistant", "Without AI assistant"])

# User uploads the JSONL file
uploaded_file = st.file_uploader("Upload JSONL file", type=['jsonl'])

# If a file has been uploaded
if uploaded_file is not None:
    df_preview = pd.read_json(uploaded_file, lines=True, nrows=5)

    # User selects the instruction and response columns
    instruction_column = st.selectbox('Select the instruction column', df_preview.columns)
    response_column = st.selectbox('Select the response column', df_preview.columns)

# Form for user input
with st.form(key='my_form'):
    # User enters the number of lines to process
    num_lines = st.number_input('Number of lines to process', min_value=1, step=1, value=5)

    # User enters the line to start processing from
    start_line = st.number_input('Start processing from line', min_value=1, step=1, value=1)

    # User enters the name of the output file
    output_file = st.text_input('Output file name', 'output.jsonl')

    # If the user chose to use the AI assistant, they enter the prompt for the assistant and select the model
    if processing_mode == "With AI assistant" and uploaded_file is not None:
        prompt = st.text_area('Enter the prompt for the assistant', 'You are a helpful coding assistant.')
        model_choice = st.selectbox('Select model', list(MODEL_MAP.keys()))
        MODEL = MODEL_MAP[model_choice]

    # User submits the form
    submit_button = st.form_submit_button(label='Submit')

# If the form is submitted
if submit_button:
    # If the user uploaded a file
    if uploaded_file is not None:
        data_list = []

        # Loop through the lines in the file
        for i, line in enumerate(uploaded_file):
            # If the current line is before the start line, skip it
            if i < start_line:
                continue
            # If we've processed the specified number of lines, stop
            if len(data_list) >= num_lines:
                break
            # Try to add the line to the data list
            try:
                record = json.loads(line)
                # If the record does not contain Chinese characters, add it to the data list
                if not contains_chinese(record[instruction_column]) and not contains_chinese(record[response_column]):
                    data_list.append(record)
            except json.JSONDecodeError:
                st.error(f"Failed to parse JSON on line {i}. Skipping this line.")

        # Convert the data list to a DataFrame
        df = pd.DataFrame(data_list)

        # Display the DataFrame
        st.write(df)

        # Create a progress bar
        progress_bar = st.progress(0)

        # If the user chose to use the AI assistant
        if processing_mode == "With AI assistant":
            for i, record in df.iterrows():
                # Update the progress bar
                progress_bar.progress((i + 1) / len(df))

                # Submit the instruction as the user message
                user_message = record[instruction_column]

                # Set system message
                messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': user_message}]

                # Prepare the data for the API request
                data = {
                    'model': MODEL,
                    'messages': messages
                }

                # Send the API request
                response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, data=json.dumps(data))

                try:
                    # Parse the response
                    response_json = response.json()

                    # Get the assistant's message from the response
                    assistant_message = response_json['choices'][0]['message']['content']

                    # Add the assistant's message to the conversation
                    messages.append({'role': 'assistant', 'content': assistant_message})

                    # Update the record with the assistant's response
                    df.at[i, response_column] = assistant_message

                    # Add the combined text field
                    df.at[i, 'text'] = combine_text(record[instruction_column], assistant_message)

                # If parsing the response fails
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON response.")

        # If the user chose not to use the AI assistant
        else:
            for i, record in df.iterrows():
                # Update the progress bar
                progress_bar.progress((i + 1) / len(df))

                # Combine the instruction and response into the "text" field
                df.at[i, 'text'] = combine_text(record[instruction_column], record[response_column])

        # Display the updated DataFrame
        st.write(df)

        # Convert DataFrame to JSONLines string
        jsonl = "\\n".join(df.apply(lambda x: x.to_json(), axis=1))

        # Encode JSONLines string to bytes, then to Base64
        b64 = base64.b64encode(jsonl.encode()).decode()

        # Create a download link for the output file
        href = f'<a href="data:file/json;base64,{b64}" download="{output_file}">Download {output_file}</a>'
        st.markdown(href, unsafe_allow_html=True)