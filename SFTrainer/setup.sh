pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
pip install -q -U datasets bitsandbytes einops scipy wandb sentencepiece
git config --global credential.helper store
huggingface-cli login
wandb login