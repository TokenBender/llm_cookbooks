
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

## Output

The script adds a new column to the dataset called 'ai_response', which contains the AI model's response for each row. The output file is a JSON file where each record is written as a separate line.

## Error Handling

If there is an error parsing the JSON response from the OpenRouter API, the script will print an error message and the HTTP response details. Processing will continue with the next row.
