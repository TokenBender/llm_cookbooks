import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb

# Initialize the Argument Parser
parser = argparse.ArgumentParser(description='Train a model with optional checkpoint resumption.')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory containing checkpoints to resume from.')
parser.add_argument('--config_file', type=str, default="config.json", help='Configuration file.')
args = parser.parse_args()

# Load Configuration File
with open(args.config_file, 'r') as f:
    config = json.load(f)

# Handle the torch.bfloat16 issue here
if config['bnb_config'].get('bnb_4bit_compute_dtype') == "torch.bfloat16":
    config['bnb_config']['bnb_4bit_compute_dtype'] = torch.bfloat16

# Get the checkpoint directory if provided
resume_from_checkpoint = args.checkpoint_dir

# Load dataset
dataset = load_dataset(config['dataset_name'], split="train")

# Load the model
bnb_config = BitsAndBytesConfig(**config['bnb_config'])
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Initialize wandb
wandb.init(project="codellama34B_qlora", name="test")

# Load training arguments
training_arguments = TrainingArguments(**config['training_args'])

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=LoraConfig(
        lora_alpha=config['lora_config']['lora_alpha'],
        lora_dropout=config['lora_config']['lora_dropout'],
        r=config['lora_config']['lora_r'],
        bias=config['lora_config']['bias'],
        task_type=config['lora_config']['task_type']
    ),
    dataset_text_field="text",
    max_seq_length=config['max_seq_length'],
    tokenizer=tokenizer,
    args=training_arguments,
)

# Additional setup for stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train the model
if resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    trainer.train()

# Save the model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained("outputs")

# Load LoRA config from saved model
lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

# Generate text (Your specific code here)
text = '''###Instruction\nGenerate a python function to print fibonacci sequence iteratively. ###Response\n'''
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
outputs = model.generate(**inputs, max_new_tokens=2048)

# Optionally push model to hub, only if enabled in config
if config['push_to_hub']['enabled']:
    model_name_for_hub = config['push_to_hub'].get('model_name', 'MyAwesomeModel')
    model.push_to_hub(model_name_for_hub)
