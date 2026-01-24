# qa.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chains.question_answering import load_qa_chain # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceHub

# ---------------------------
# 1️⃣ Load your FAISS vectorstore
# ---------------------------
# Path to your FAISS index folder
faiss_index_path = os.path.join("data", "faiss_index")

# HuggingFace embedding model (local, free, no quota)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
db = FAISS.load_local(faiss_index_path, embedding_function=embedding_function)

# ---------------------------
# 2️⃣ Set up retriever
# ---------------------------
retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------------------------
# 3️⃣ Set up LLM for answering questions
# ---------------------------
# You can use HuggingFaceHub for local inference, or any other local model
# Replace your repo_id with the HuggingFace model you want
llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 512})

# QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# ---------------------------
# 4️⃣ Function to answer questions
# ---------------------------
def answer_question(question: str) -> str:
    """
    Returns the answer for a given question using local FAISS + HuggingFace embeddings.
    """
    # Retrieve top relevant docs from FAISS
    docs = retriever.get_relevant_documents(question)

    # Run QA chain on retrieved docs
    answer = qa_chain.run(input_documents=docs, question=question)
    return answer
