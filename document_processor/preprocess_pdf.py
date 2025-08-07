import json
from utils import extract_text_from_pdf, chunk_text

# Path to your PDF
pdf_path = "sample.pdf"  # 👈 Change this to match your actual PDF file name

# Step 1: Extract text from PDF
text = extract_text_from_pdf(pdf_path)

# Step 2: Chunk the text
chunks = chunk_text(text)

# Step 3: Save chunks to a JSON file
chunk_dicts = [{"text": chunk} for chunk in chunks]  # match main.py format

with open("parsed_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(chunk_dicts)} chunks to parsed_chunks.json")

