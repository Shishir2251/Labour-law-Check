import os 
from langchain_community.vectorsstore import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.utils.loaders import load_document
from app.utils.text_splitter import split_documents

VECTOR_PATH ="data/vectorstore"

def ingest_documents(file_paths: str):
    documents = load_document(file_path)
    chunks = split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    if os.path.exists(VECTOR_PATH):
        db = FAISS.load_local(VECTOR_PATH, embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(VECTOR_PATH)
