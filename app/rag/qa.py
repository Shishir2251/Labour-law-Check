from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from app.core import config

VECTORSTORE_PATH = config.VECTORSTORE_PATH

def answer_question(question: str):
    # Local embeddings (no API)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    # HuggingFace LLM
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        temperature=0.2,
        max_new_tokens=256
    )

    prompt = f"""
You are a legal assistant for Bangladesh Labour Law.
Answer strictly from the context.
If not found, say: Not found in Bangladesh Labour Law.

Context:
{context}

Question:
{question}
"""

    return {"answer": llm.invoke(prompt)}
