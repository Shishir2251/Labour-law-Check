from langchain_google_genai import ChatGoogleGenerativeAI
from app.rag.retriever import get_retriever

def answer_question(question: str):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = f"""
You are a legal assistant for Bangladesh Labour Law.
Answer strictly from the context.
If the answer is not in the context, say "Not found in Bangladesh Labour Law".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}
