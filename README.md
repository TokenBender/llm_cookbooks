# llm_cookbooks
LLM Anarchist's Cookbook

Focus on freedom of usage of LLMs as well as freeing the LLMs from RLHF lobotomy

[20/07/23]

Started prepping my RAG fine-tuning setup.

Going to fine-tune llama2 for retrieval augmented generation.

Going to call it RAGA (राग in Hindi, means Melody) 

RAG isn't simple as it seems and there are many rules to follow and many things to care about:
* Extracting correct answer based on a given context
* Extracting correct answer based on question when there are multiple docs fetched in the context
* Answering in negative when there is no answer fetched in the context
* Answering question by inference and combining multiple fetched docs to answer the question that can't be answered directly.

TODO:
- [] ~~Test fine-tune llamav2 with 10 rows of SQuAD v2 to see if the fine tuning pipeline is working (Criteria: Can be used anywhere with 4bit, 8bit, full fine-tune)~~
    - [] ~~Test pipeline by Younes Belkada for guanaco fine-tuning~~
      - [] ~~The training losses drop to zero after 10 steps for some reason I don't understand~~
    - [] ~~Test scale llm fine tuning library~~
    - [] ~~Trying this out - Test Autotrain by HF for fine-tuning llama~~
      - [] ~~Facing issue with autotrain advanced installation~~
    - [] ~~Test Philip Scmidt's Amazon Sagemaker guide to fine-tune~~

I tweaked existing code_instructions_120k dataset on HF to match alpaca style dataset.

[21/07/23]

[x] Find a fine tune guide that works

So I made something useful today and that is a reward in itself.
I prepared a 122k alpaca style coding instruction dataset and fine-tuned the model with it for 200 steps.
As a result, it seems to be doing very well.

I've a few things in mind and after that this will be more valuable.
- [] Add a chat UI
- [] Creating a merged model, currently the repo contains lora adapters that will need to loaded on top of Meta's llama2 so access gating is an issue
- [] I'll quantize these, possibly tonight or tomorrow in the day, then it can be run locally with 4G ram
- [] I've used alpaca style instruction tuning, I'll switch to llama2 style [INST]<<SYS>> style and see if it improves anything
- [] HumanEval report and checking for any training data leaks
- [] I'll try 8k context via RoPE enhancement as well, let's see if that degrades performance or not.