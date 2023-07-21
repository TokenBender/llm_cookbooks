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
- [] Test fine-tune llamav2 with 10 rows of SQuAD v2 to see if the fine tuning pipeline is working (Criteria: Can be used anywhere with 4bit, 8bit, full fine-tune)
    - [] Test pipeline by Younes Belkada for guanaco fine-tuning
      - [] The training losses drop to zero after 10 steps for some reason I don't understand
    - [] Test scale llm fine tuning library
    - [] Trying this out - Test Autotrain by HF for fine-tuning llama
      - [] Facing issue with autotrain advanced installation
    - [] ~~Test Philip Scmidt's Amazon Sagemaker guide to fine-tune~~

I tweaked existing code_instructions_120k dataset on HF to match alpaca style dataset.

[21/07/23]

