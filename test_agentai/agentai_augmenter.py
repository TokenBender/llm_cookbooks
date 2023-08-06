import json
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# Set up the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_variations(prompt, completion, num_variations=5):
    """
    Generate variations of a given prompt and completion using the OpenAI GPT-3 API.

    Args:
        prompt (str): The original prompt.
        completion (str): The original completion.
        num_variations (int): The number of variations to generate.

    Returns:
        list: A list of variations in json format with prompt and completions as keys.
    """
    variations = []

    for _ in range(num_variations):
        # Create a series of chat messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
            {"role": "user", "content": "Generate a similar prompt and completion."},
        ]

        # Use the OpenAI GPT-3 API to generate a variation of the prompt and completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=120,
        )

        # Extract the new prompt and completion from the response
        new_prompt_and_completion = response['choices'][0]['message']['content']

        # Add the new prompt and completion pair to the variations
        variations.append({'prompt_and_completion': new_prompt_and_completion})

    return variations

def process_jsonl_file(input_file, output_file):
    """
    Process a JSONL file to generate variations of prompts and completions.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSONL file.
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Parse the JSON line to extract the prompt and completion
            data = json.loads(line)
            prompt = data['prompt']
            completion = data['canonical_solution']

            # Generate variations of the prompt and completion
            variations = generate_variations(prompt, completion)

            # Save the augmented data back to a JSONL file
            for variation in variations:
                f_out.write(json.dumps(variation) + '\n')

# Call the function to process the JSONL file
process_jsonl_file('Humaneval.jsonl', 'Humaneval_augmented.jsonl')