# Solution Steps

1. Import required libraries: LangChain, OpenAI, Chroma, tiktoken, numpy, time, and CSV for logging.

2. Configure model names, embedding dimension, retrieval parameters (TOP_K), context token budget, and the Chroma connection with matching embedding model.

3. Implement a function to encode input queries as embedding vectors using OpenAIEmbeddings. 

4. Define a cosine similarity computation for robust post-filtering of Chroma retrieval results based on the actual embedding vectors (assumes embeddings are stored in doc metadata). 

5. Implement 'retrieve_faq_chunks' that retrieves FAQ documents from Chroma, with category filtering (via Chroma filter) if supplied, and returns top chunks via cosine similarity to the query embedding.

6. Write utilities for counting tokens in strings using tiktoken; also implement logic to trim list of context chunks to fit token budget constraints.

7. Build the context section of the prompt with citation markers ([docN]), ensuring only the amount of retrieved context that fits within CONTEXT_TOKEN_BUDGET is included.

8. Assemble the final prompt: insert a few-shot example, the current question, label for the answer, and the assembled context (with citations).

9. Implement the evaluation/logging infrastructure: log question, latency, retrieved count, token stats, precision@k, category, and the used citations for monitoring performance.

10. Write a simple 'precision@k' calculation: measures if query terms appear in each retrieved chunk, for rough recall estimate.

11. Implement the core RAG pipeline function ('rag_answer'), which ties together query encoding, FAQ retrieval, context prompt construction, prompt assembly, GPT call, and logging. Return all relevant results.

12. Add __main__ usage/CLI example: run rag_answer with a test question and category, show outputs.

