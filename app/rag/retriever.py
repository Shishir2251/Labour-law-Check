from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core import config

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=config.GOOGLE_API_KEY
)


    db = FAISS.load_local(
        config.VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 4})

