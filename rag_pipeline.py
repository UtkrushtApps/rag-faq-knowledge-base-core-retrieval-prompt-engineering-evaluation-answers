import time
import csv
import openai
from typing import List, Optional, Dict
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import tiktoken
import numpy as np

# ---- Configuration ----
EMBEDDING_MODEL = 'text-embedding-ada-002'
GPT_COMPLETION_MODEL = 'gpt-3.5-turbo'
EMBEDDING_DIM = 1536
TOP_K = 6  # Chunks to fetch
CONTEXT_TOKEN_BUDGET = 2500  # max tokens for context (rest for question+prompt, fits gpt3.5-turbo-window)
CSV_LOGFILE = 'rag_eval_log.csv'
CATEGORY_FIELD_NAME = 'category'  # on docs' metadata

# ---- Setup: Assume Chroma has already ingested the docs with embedding model exactly matching EMBEDDING_MODEL ----
chroma = Chroma(collection_name="faq_collection", embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
llm = OpenAI(temperature=0, model_name=GPT_COMPLETION_MODEL, max_tokens=512)
tokenc = tiktoken.encoding_for_model(GPT_COMPLETION_MODEL)

# --- Few-shot prompt example ---
FEWSHOT_EXAMPLE = (
    'Q: How do I reset my account password?\n'
    'Helpful Answer: To reset your account password, visit the login page, select "Forgot Password", and follow the instructions. [doc1]\n'
)

# --- Main: Retrieval and LLM pipeline ---
def compute_cosine_similarity(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
    # Ensure 1d arrays
    return float(np.dot(q_emb, d_emb) / ((np.linalg.norm(q_emb) + 1e-10) * (np.linalg.norm(d_emb) + 1e-10)))

def encode_query(query: str) -> np.ndarray:
    # OpenAIEmbeddings' embed_query returns a list
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL).embed_query(query)
    return np.array(emb)


def retrieve_faq_chunks(query: str, category: Optional[str]=None, k: int=TOP_K) -> List[Document]:
    
    # Create filter if category is given
    chroma_filter = None
    if category:
        chroma_filter = {CATEGORY_FIELD_NAME: category}
    
    # Use Chroma's similarity_search_with_score (returns [(Document, score)])
    docs_with_scores = chroma.similarity_search_with_score(query, k=3*TOP_K, filter=chroma_filter)
    # (Extra docs to allow us to post-filter for cosine similarity accuracy)
    
    # Compute embedding for query for custom cosine scoring
    query_emb = encode_query(query)
    filtered = []
    for doc, chroma_score in docs_with_scores:
        doc_emb = np.array(doc.metadata.get('embedding', None))
        if doc_emb is None:
            # Retrieve embedding via start (should ideally never happen if pre-ingested correctly)
            continue
        sim = compute_cosine_similarity(query_emb, doc_emb)
        filtered.append((doc, sim))
    filtered = sorted(filtered, key=lambda x:x[1], reverse=True)
    return [doc for doc, sim in filtered[:k]]


def count_tokens(text: str) -> int:
    return len(tokenc.encode(text))


def trim_context_for_token_budget(context_chunks: List[str], budget: int) -> List[str]:
    result = []
    total = 0
    for chunk in context_chunks:
        n = count_tokens(chunk)
        if total + n > budget:
            break
        result.append(chunk)
        total += n
    return result


def build_context_prompt(question: str, docs: List[Document]) -> (str, List[int]):
    # Each doc will be cited as [docN] where N is 1-based idx in this batch
    context_texts = []
    citations = []
    for i, doc in enumerate(docs):
        cite = f"[doc{i+1}]"
        # Ensure passage is plain text (not dict)
        passage = doc.page_content.strip()
        # Optionally, could limit by sentence count or passage len
        context_chunk = f"{passage} {cite}"
        context_texts.append(context_chunk)
        citations.append(i+1)
    # Now trim to fit context budget
    context_texts = trim_context_for_token_budget(context_texts, CONTEXT_TOKEN_BUDGET)
    prompt_context = '\n'.join(context_texts)
    return prompt_context, citations


def build_final_prompt(question: str, context: str) -> str:
    prompt = (
        FEWSHOT_EXAMPLE +
        f"Q: {question}\n"
        f"Helpful Answer (include only info relevant to the question and cite like [docN]):\n"
        f"Context:\n{context}\n"
        f"A:"
    )
    return prompt

# --- Evaluation logging ---
def write_log(row: Dict):
    # Append to evaluation csv
    fieldnames = [
        'question', 'latency', 'retrieved_chunks', 'context_tokens',
        'total_tokens', 'precision_at_k', 'category', 'used_citations', 'gpt_usage_tokens'   # 'gpt_usage_tokens' to help track OpenAI billing
    ]
    is_new = False
    try:
        with open(CSV_LOGFILE, 'r', encoding='utf8') as f:
            pass
    except FileNotFoundError:
        is_new = True
    with open(CSV_LOGFILE, 'a', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def evaluate_precision_at_k(retrieved_docs: List[Document], query: str) -> float:
    # For simplicity: if at least one doc contains an answer to the query, count as precision 1, else 0
    # In reality, you might have ground truth docids, but here we search if query key words hit in chunk
    query_terms = set(query.lower().split())
    relevant_hits = 0
    for doc in retrieved_docs:
        content = doc.page_content.lower()
        if any(t in content for t in query_terms):
            relevant_hits += 1
    return relevant_hits / len(retrieved_docs) if retrieved_docs else 0

# ---- Core pipeline API ----
def rag_answer(question: str, category: Optional[str]=None) -> Dict:
    start = time.time()
    retrieved_docs = retrieve_faq_chunks(question, category)
    context, used_citations = build_context_prompt(question, retrieved_docs)
    prompt = build_final_prompt(question, context)

    context_tokens = count_tokens(context)
    total_tokens = count_tokens(prompt)
    
    # Call LLM (OpenAI)
    llm_response = llm(prompt)
    latency = time.time() - start
    precision_at_k = evaluate_precision_at_k(retrieved_docs, question)
    
    # Estimate token usage reported by OpenAI (will be correct only if using langchain's OpenAI LLM wrapper)
    response_tokens = count_tokens(llm_response)
    gpt_usage_tokens = total_tokens + response_tokens

    # Log
    write_log({
        'question': question,
        'latency': latency,
        'retrieved_chunks': len(retrieved_docs),
        'context_tokens': context_tokens,
        'total_tokens': total_tokens,
        'precision_at_k': precision_at_k,
        'category': category or '',
        'used_citations': ','.join([f'doc{n}' for n in used_citations]),
        'gpt_usage_tokens': gpt_usage_tokens
    })
    
    return {
        'answer': llm_response,
        'context': context,
        'retrieved_docs': retrieved_docs,
        'latency': latency,
        'precision_at_k': precision_at_k,
        'used_citations': used_citations
    }

if __name__ == "__main__":
    # Example usage
    q = "How can I update my billing address?"
    result = rag_answer(q, category="billing")
    print("\n---\nRAG pipeline output:")
    print("Answer:", result['answer'])
    print("Context used:\n", result['context'])
    print("Retrieval latency: %.3fs" % result['latency'])
    print("Precision@k: %.2f" % result['precision_at_k'])
