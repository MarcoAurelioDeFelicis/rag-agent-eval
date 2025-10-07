from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def create_rag_agent(db):

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

    '''PROMPT TEMEPLATE'''
    prompt = ChatPromptTemplate.from_template("""
You are an exper chef assistant. Answer the user's question in a helpful and concise manner, 
based solely on the following context:
<context>
{context}
</context>
Question: {input}
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    # Combined chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain