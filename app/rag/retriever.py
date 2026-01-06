from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from app.core import config

VECTORSTORE_DIR = Path(__file__).resolve().parents[2] / "data" / "vectorstore"
INDEX_NAME = "labour_law_index"

def get_retriever():
    embeddings = GooglePalmEmbeddings(api_key=config.GOOGLE_API_KEY)
    db = FAISS.load_local(str(VECTORSTORE_DIR / INDEX_NAME), embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever
