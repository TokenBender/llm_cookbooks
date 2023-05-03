import os
from dotenv import load_dotenv

def get_api_key_from_env_file():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def get_api_key_from_os_env():
    return os.getenv("OPENAI_API_KEY")

def get_api_key(api_key_method):
    return api_key_method()