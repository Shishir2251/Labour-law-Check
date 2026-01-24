# build_vectorstore.py

import os
from PyPDF2 import PdfReader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# 1️⃣ Path to folder containing your PDF files
docs_folder = r"C:\Users\Shishir\Projects\labour-law-rag\data\raw"
vectorstore_path = r"C:\Users\Shishir\Projects\labour-law-rag\data\faiss_index"

# 2️⃣ Initialize local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3️⃣ Load and combine text from all PDFs
all_texts = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(docs_folder, filename)
        print(f"Processing {file_path}...")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        all_texts.append(text)

# 4️⃣ Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

docs = text_splitter.split_text("\n".join(all_texts))

# 5️⃣ Build FAISS vector store
print("Building FAISS index...")
db = FAISS.from_texts(docs, embeddings)

# 6️⃣ Save locally
db.save_local(vectorstore_path)
print(f"FAISS index saved at: {vectorstore_path}")
