from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from app.rag.retriever import get_retriever

def answer_question(question: str):
    retriever = get_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2
    )

    system_prompt = """
    You are a legal AI assistant specialized in Bangladesh Labour Law.
    Answer strictly based on the provided legal context.
    If the information is not found in the law, say clearly that it is not specified.
    """

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": None
        }
    )

    result = qa.invoke({"query": question})
    return result
