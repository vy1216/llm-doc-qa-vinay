'''
import json
import openai
from dotenv import load_dotenv
load_dotenv()
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_pdf, chunk_text

# Load a sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy OpenAI API key - replace with your own
openai.api_key = "sk-REPLACE_WITH_YOUR_KEY"

# Load parsed document chunks (you can modify this to use real PDF files)
with open("parsed_chunks.json", "r") as f:
    chunks = json.load(f)

# Compute embeddings
chunk_texts = [c["text"] for c in chunks]
embeddings = model.encode(chunk_texts)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def answer_query(query):
    top_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    prompt = f"""
You are a helpful assistant for insurance document processing.
Based on the following document clauses:

{context}

Answer the following query:
"{query}"

Return your response in JSON format with keys: decision, amount (if applicable), and justification.
"""

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
)



    

    return response.choices[0].message["content"]

# --- Try the system ---
query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
result = answer_query(query)
print(result) '''

import json
import openai
from dotenv import load_dotenv
load_dotenv()
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_pdf, chunk_text

# Load a sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load OpenAI API key from environment (optional if using real API later)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load parsed document chunks
with open("parsed_chunks.json", "r") as f:
    chunks = json.load(f)

# Compute embeddings for chunks
chunk_texts = [c["text"] for c in chunks]
embeddings = model.encode(chunk_texts)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def answer_query(query, use_mock=True):
    top_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    prompt = f"""
You are a helpful assistant for insurance document processing.
Based on the following document clauses:

{context}

Answer the following query:
"{query}"

Return your response in JSON format with keys: decision, amount (if applicable), and justification.
"""

    # ✅ Use mock response if no quota available
    if use_mock:
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "decision": "approved",
                        "amount": "₹50,000",
                        "justification": "Surgery is covered under policy terms. Waiting period has passed."
                    })
                }
            }]
        }
        return mock_response["choices"][0]["message"]["content"]

    # 🔁 If you have quota, use OpenAI API
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message["content"]

# --- Try the system ---
query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
result = answer_query(query, use_mock=True)  # Set to False when you regain OpenAI quota
print(result)

