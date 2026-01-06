from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.rag.retriever import get_retriever


def answer_question(question: str):
    retriever = get_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a legal AI assistant specialized in Bangladesh Labour Law.
        Answer ONLY using the context provided.
        If the answer is not found in the law, say clearly:
        "This information is not specified in Bangladesh Labour Law."

        Context:
        {context}

        Question:
        {question}
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    response = rag_chain.invoke(question)

    return {
        "answer": response.content
    }
