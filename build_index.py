from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

PDF_PATH = "data/raw/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf"  # your PDF file name
DB_PATH = "data/vectorstore"

os.makedirs(DB_PATH, exist_ok=True)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(chunks, embeddings)
db.save_local(DB_PATH)

print("✅ Vector store created successfully!")
