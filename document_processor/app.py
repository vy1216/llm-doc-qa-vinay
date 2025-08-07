import streamlit as st
import fitz  # PyMuPDF
import json
import os
import openai

# Set your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly: "sk-..." (Not recommended to hardcode)

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 LLM-Powered Document Q&A")

st.markdown("Upload one or more PDF files and ask questions based on their content.")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Parse and chunk text
def extract_chunks_from_pdf(file, chunk_size=500, overlap=50):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Chunking
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[i:i + chunk_size]
        chunks.append({"content": chunk})
    return chunks

all_chunks = []

if uploaded_files:
    for file in uploaded_files:
        chunks = extract_chunks_from_pdf(file)
        all_chunks.extend(chunks)
    st.success(f"✅ Processed {len(uploaded_files)} PDF(s), got {len(all_chunks)} chunks.")
    
    # Optionally save to JSON
    with open("parsed_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
else:
    st.warning("Please upload at least one PDF.")

# Q&A Section
st.subheader("🔍 Ask a question")

question = st.text_input("Enter your question")

if st.button("Get Answer") and question and all_chunks:
    # Select top N chunks as context
    top_chunks = "\n".join([chunk["content"] for chunk in all_chunks[:10]])

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant helping users understand PDF documents."},
                {"role": "user", "content": f"Context:\n{top_chunks}\n\nQuestion: {question}"}
            ]
        )
        answer = response['choices'][0]['message']['content']
        st.markdown("### 🧠 Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"⚠️ API call failed: {e}")

elif question:
    st.info("Please upload and process PDF documents before asking.")


