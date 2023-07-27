from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import filter_code, run_eval, fix_indents
import torch

# add hugging face access token here
TOKEN = "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"

def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        top_k=10,
        do_sample=True,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/llama2/eval.jsonl"
    os.makedirs("results/llama2", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "TokenBender/llama2-7b-chat-hf-codeCherryPop-qLoRA-merged",
    )

    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            "TokenBender/llama2-7b-chat-hf-codeCherryPop-qLoRA-merged",
            torch_dtype=torch.float16,
        )
        .eval()
        .to("cuda:0")
    )

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )