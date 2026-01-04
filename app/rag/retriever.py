from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

VECTOR_PATH = "data/vectorstore"

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    db = FAISS.load_local(VECTOR_PATH, embeddings)
    return db.as_retriever(search_kwargs={"k": 4})
