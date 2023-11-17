import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, is_xpu_available
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser
import yaml

# Function to load configuration from a YAML file
def load_config_from_yaml(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to initialize and return the model
def initialize_model(config):
    if config.get('load_in_8bit') and config.get('load_in_4bit'):
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")

    quantization_config = None
    if config.get('load_in_8bit') or config.get('load_in_4bit'):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=config.get('load_in_8bit', False),
            load_in_4bit=config.get('load_in_4bit', False)
        )

    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16 if quantization_config else None

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=config.get('trust_remote_code', False),
        torch_dtype=torch_dtype,
        use_flash_attention_2=config.get('use_flash_attention2', False),
        use_auth_token=config.get('use_auth_token', True),
    )

    return model

# Function to create and return training arguments
def create_training_args(config):
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        logging_steps=config['logging_steps'],
        optim=config['optimizer'],
        num_train_epochs=config['num_train_epochs'],
        max_steps=config.get('max_steps', -1),
        report_to=config.get('log_with', 'none'),
        save_steps=config['save_steps'],
        save_total_limit=config.get('save_total_limit', None),
        push_to_hub=config.get('push_to_hub', False),
        hub_model_id=config.get('hub_model_id', None),
        gradient_checkpointing=config.get('gradient_checkpointing', False),
        # Add more arguments as needed from your configuration
    )
    return training_args

# Load configuration from the YAML file
config_path = 'config.yml'  # Replace with your YAML file path
config = load_config_from_yaml(config_path)

# Initialize the model
model = initialize_model(config)

# Load the dataset
dataset = load_dataset(config['dataset_name'], split="train")

# Create training arguments
training_args = create_training_args(config)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=config['seq_length'],
    train_dataset=dataset,
    dataset_text_field=config['dataset_text_field'],
    # Define peft_config if use_peft is True in your configuration
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(config['output_dir'])