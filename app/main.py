from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.qa import answer_question

app = FastAPI(title="Bangladesh Labour Law AI (Gemini)")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_law(query: Query):
    result = answer_question(query.question)
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
