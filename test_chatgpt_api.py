from chatgpt_api import generate_response_chatgpt
import api_key_handler

def get_multiline_input(prompt):
    print(prompt, end="")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    # Get the API key using the desired method
    api_key = api_key_handler.get_api_key(api_key_handler.get_api_key_from_env_file)
    
    question = get_multiline_input("Please enter your question (press Enter twice to finish):\n")

    # Call the ChatGPT model
    chatgpt_response = generate_response_chatgpt(api_key, question)
    print("\nChatGPT Model Response:")
    print(chatgpt_response)

if __name__ == "__main__":
    main()