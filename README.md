# llm_cookbooks
Lightweight &amp; fast implementations of LLM  APIs

[20/07/23]
Started prepping my RAG fine-tuning setup.

Going to fine-tune llama2 for retrieval augmented generation.

Going to call it RAGA (राग in Hindi, means Melody) 

RAG isn't simple as it seems and there are many rules to follow and many things to care about:
* Extracting correct answer based on a given context
* Extracting correct answer based on question when there are multiple docs fetched in the context
* Answering in negative when there is no answer fetched in the context
* Answering question by inference and combining multiple fetched docs to answer the question that can't be answered directly.
