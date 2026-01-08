import os
from pathlib import Path

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import GooglePalmEmbeddings  # Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core import config  # Make sure config.py loads .env

# Paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME = "labour_law_index"

def load_pdfs(pdf_folder: Path):
    docs = []
    for pdf_file in pdf_folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    print(f"📄 Loaded {len(docs)} documents from {pdf_folder}")
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=config.GOOGLE_API_KEY
)

    
    # Build FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save locally
    vectorstore.save_local(str(VECTORSTORE_DIR / INDEX_NAME))
    print(f"💾 Vectorstore saved to {VECTORSTORE_DIR}")
    return vectorstore

if __name__ == "__main__":
    raw_docs = load_pdfs(RAW_DIR)
    chunks = split_docs(raw_docs)
    create_vectorstore(chunks)
