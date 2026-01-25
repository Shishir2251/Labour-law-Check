# app/rag/qa.py

from pathlib import Path
import subprocess
import json

# Correct imports for latest LangChain
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# -------------------------------
# 1️⃣ Embeddings & Vectorstore
# -------------------------------
# Using a local sentence-transformers model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTORSTORE_PATH = Path("data/vectorstore/labour_law_index")
if VECTORSTORE_PATH.exists():
    vectorstore = FAISS.load_local(str(VECTORSTORE_PATH), embeddings_model,allow_dangerous_deserialization=True)
else:
    # Initialize empty FAISS if index does not exist
    vectorstore = FAISS(embedding_function=embeddings_model, index=None)

# -------------------------------
# 2️⃣ Helper: Run Ollama locally
# -------------------------------
def run_ollama(prompt: str) -> str:
    """
    Sends the prompt to LLaMA3 via Ollama CLI.
    """
    
        
    ollama_path=r"C:\Users\Shishir\AppData\Local\Programs\Ollama\ollama.exe"
    model_name = "llama3"
    try:
        result = subprocess.run(
            [ollama_path,"run",model_name, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error running Ollama: {e.stderr.strip()}"
    except PermissionError:
        return "Permission denied: Cannot execute Ollama. Try running your Python/terminal as Administrator."

 


# -------------------------------
# 3️⃣ Prompt Template
# -------------------------------
qa_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal assistant specialized in Bangladesh labour law.

Use the context below to answer the question as accurately as possible.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------
# 4️⃣ Main function
# -------------------------------
def answer_question(question: str) -> dict:
    # Step 1: Retrieve relevant docs from vectorstore
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Step 2: Format prompt
    prompt = qa_template.format(context=context, question=question)

    # Step 3: Run local LLM
    answer = run_ollama(prompt)

    return {"answer": answer}
