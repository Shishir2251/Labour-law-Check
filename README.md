# Labour Law RAG - FastAPI Application

A **Retrieval-Augmented Generation (RAG)** system for Bangladesh Labour Law, built using **FastAPI**, **FAISS**, and **Ollama** for language model queries. This application allows users to ask questions related to labour law and get context-aware answers using a local or cloud-hosted language model.

---

## 📝 Features

- Ask questions related to Bangladesh Labour Law.
- Uses **FAISS** vector store for fast document retrieval.
- Embeddings generated via **HuggingFaceEmbeddings**.
- Integrates **Ollama** LLM to generate answers.
- API endpoints with **Swagger UI** support.
- Ready for local testing and potential cloud deployment.

---

## ⚡ Requirements

- Python 3.11+
- Ollama (for running local LLMs)
- Git (for cloning the repository)
- Windows / Linux / MacOS

**Python packages:**
```bash
pip install -r requirements.txt

Key packages used:

fastapi

uvicorn

langchain

langchain-huggingface

faiss-cpu

pydantic

📂 Project Structure
labour-law-rag/
│
├─ app/
│   ├─ main.py            # FastAPI entrypoint
│   ├─ rag/
│   │   ├─ qa.py          # Core RAG + Ollama logic
│   │   └─ ... 
│   └─ ...
├─ VECTORSTORE/           # FAISS vectorstore files
├─ requirements.txt
└─ README.md

Local Setup

1.Clone the repository:
git clone https://github.com/<your-username>/labour-law-rag.git
cd labour-law-rag
2.Create a virtual environment:
python -m venv lenv
lenv\Scripts\activate      # Windows
source lenv/bin/activate   # Linux/Mac
3.Install dependencies:
pip install -r requirements.txt
4.Make sure Ollama is installed and accessible in your PATH.
Test with:
ollama list
5.Run the FastAPI app:
uvicorn app.main:app --reload
6.Open in browser:

Swagger docs: http://127.0.0.1:8000/docs

OpenAPI JSON: http://127.0.0.1:8000/openapi.json

API Endpoints
POST /ask

Ask a question and get an answer from the model.

Request:
{
  "question": "What are the working hour regulations in Bangladesh?"
}
Response:
{
  "answer": "According to Bangladesh Labour Law, ..."
}
Configuration

Vectorstore path:
By default, the FAISS vector store is stored in VECTORSTORE/.

Embedding model:
Uses HuggingFace embeddings:
from langchain_huggingface import HuggingFaceEmbeddings
llama model:
Update your model in qa.py:
MODEL_NAME = "llama3"  # or any model you pulled
Notes

1.Ollama must be installed locally for the app to work.

2.If deploying to the cloud, replace Ollama with API-based LLM (like OpenAI) because free cloud providers cannot run desktop apps.

3.FAISS deserialization may require allow_dangerous_deserialization=True if loading pickle files.

Troubleshooting

422 Unprocessable Entity → Make sure request body is JSON with question key.

500 Internal Server Error → Check if Ollama executable is in PATH and vectorstore is loaded.

PermissionError (WinError 5) → Run terminal as Administrator on Windows.

FAISS ValueError → Use allow_dangerous_deserialization=True if loading your own pickled vectorstore.

Contribution

Contributions are welcome!
Feel free to submit issues, improvements, or new features.


---

I can also make a **shorter “deploy-ready” README** with **direct GitHub + Render/Railway instructions**, so someone can just click and run your app in the cloud.  

Do you want me to create that version too?
