# Fine-Tuning Configuration and Script Explanation

This README provides a detailed explanation of the fine-tuning script and the accompanying `config.yml` configuration file.

## Installation
Run setup.sh, it will take care of setting HF, wandb api logins and other dependencies for installation

## Overview

The script is designed to fine-tune a pre-trained causal language model (CLM) using the SFTTrainer from the Transformers library. The process involves loading a pre-trained model, preparing a dataset, setting up training arguments, and running the training process. The `config.yml` file is used to specify all the parameters required for fine-tuning.

## Configuration File: `config.yml`

The `config.yml` file contains all the necessary configuration parameters for the fine-tuning process. Here's a breakdown of each parameter within the file:

### Model Configuration

- `model_name`: Specifies the pre-trained model to fine-tune. The default is "facebook/opt-350m", but you can replace this with the name of any model available on the Hugging Face Model Hub.

### Dataset Configuration

- `dataset_name`: The name of the dataset to use for training. The script expects this dataset to be accessible via the Hugging Face Datasets library.
- `dataset_text_field`: The key in the dataset that contains the textual data for training.

### Training Configuration

- `learning_rate`: The step size used by the optimizer during training. Affects how quickly or slowly the model learns.
- `batch_size`: Number of examples processed in one forward/backward pass. A larger batch size requires more memory.
- `seq_length`: The length of the input sequences to the model. This should be set based on the maximum context size the model can handle.
- `gradient_accumulation_steps`: Used to accumulate gradients over multiple steps to effectively increase the batch size without increasing the memory usage.
- `num_train_epochs`: The number of times the training process will iterate over the entire dataset.
- `max_steps`: An alternative to epochs; this specifies the total number of training steps. Training will stop once this number is reached.

### Hardware Configuration

- `load_in_8bit`: When set to true, the model is loaded in 8-bit precision to save memory, potentially at the cost of model precision.
- `load_in_4bit`: Similar to the above, but with 4-bit precision for even more memory savings.

### Logging and Saving

- `logging_steps`: Determines how often to log training metrics. A lower number means more frequent logging.
- `save_steps`: Specifies how often to save model checkpoints.
- `save_total_limit`: The maximum number of checkpoints to keep.

### PEFT Configuration

- `use_peft`: Indicates whether to use Parameter-Efficient Fine-Tuning (PEFT) for training adapters.
- `peft_lora_r`: The rank hyperparameter for LoRA adapters if PEFT is used.
- `peft_lora_alpha`: The scale hyperparameter for LoRA adapters.

### Security and Access

- `trust_remote_code`: If set to true, allows the execution of remote code when loading models, which can be a security risk.
- `use_auth_token`: Indicates whether to use a Hugging Face authentication token for accessing private models or datasets.

### Output and Hub Integration

- `output_dir`: The directory where the fine-tuned model and outputs will be saved.
- `push_to_hub`: If true, the fine-tuned model will be pushed to the Hugging Face Hub.
- `hub_model_id`: The model ID for the model on the Hugging Face Hub.

### Advanced Configuration

- `gradient_checkpointing`: Enables gradient checkpointing to save memory.
- `gradient_checkpointing_kwargs`: Additional keyword arguments for `torch.utils.checkpoint.checkpoint`.

## Script Functionality

The script works as follows:

1. **Load Model**: It loads the pre-trained model specified in the `config.yml`.
2. **Prepare Dataset**: The dataset specified in the `config.yml` is loaded and prepared for training.
3. **Set Training Arguments**: Training arguments, such as batch size, learning rate, and device setup, are configured.
4. **Train Model**: The model is fine-tuned using the SFTTrainer, which handles the training loop, optimization, and logging.
5. **Save Model**: After training, the model is saved to the specified `output_dir`.

## Execution

To execute the fine-tuning process, run the script with the `config.yml` file in the same directory. The script will read the parameters from the `config.yml` and perform the fine-tuning according to the specified configuration.
