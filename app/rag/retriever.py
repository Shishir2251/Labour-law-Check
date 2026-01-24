# retriever.py
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from app.core import config

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        config.VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 4})
