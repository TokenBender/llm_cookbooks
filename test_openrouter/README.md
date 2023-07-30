
# AI Augmentation Script

This project uses an AI model to augment or clean up data in a CSV or JSON dataset.

## Requirements

This script requires Python 3.6 or later. Before running the script, you'll need to install several Python packages:

```
pip install pandas requests python-dotenv
```

You will also need to obtain an API key from the OpenRouter service.

## Configuration

Create a `.env` file in the same directory as the script, and add your OpenRouter API key and your site URL:

```
OPENROUTER_API_KEY=your-api-key
YOUR_SITE_URL=your-site-url
```

## Running the Script

To run the script, use the following command:

```
python augment_script.py
```

Replace `augment_script.py` with the name of the script file.

The script will process data in batches of 4 rows at a time and will store the results in a new JSON file. The name of the output file will be the same as the input file, but with '_augmented' appended.

The AI model used is specified by the `MODEL_ACRONYM` variable in the script. By default, this is set to 'GPT4', but you can change it to use a different model. The available models are listed in the `MODEL_MAP` dictionary in the script.

## Input Columns and Processing Function

The script processes the data based on the `input_columns` list and the `process_row` function defined in the script.

`input_columns` is a list of column names that will be processed by the `process_row` function. For example:

```python
input_columns = ['column1', 'column2']
```

`process_row` is a function that takes a row of data and returns a dictionary of `{column_name: new_value}` pairs. The new values are used to update the corresponding columns in the row. You should modify this function to suit your specific use case.

For example, the following `process_row` function uses an AI model to generate a response for `column1`:

```python
def process_row(row):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant designed to serve the user.'},
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

        return {'column1': assistant_message}

    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        print("Status code:", response.status_code)
        print("Response text:", response.text)
        return {}
```

In this function, an AI model generates a response for `column1`, and the function returns a dictionary with the new value for `column1`.

## Output

The script adds new values to the specified input columns based on the processing function. The output file is a JSON file where each record is written as a separate line.

## Error Handling

If there is an error parsing the JSON response from the OpenRouter API, the script will print an error message and the HTTP response details. Processing will continue with the next row.
