from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


'''RETRIVAL AUGMENTED GENERATION'''

def create_rag_agent(db, model_name: str):

    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.7
    )

    '''PROMPT TEMEPLATE'''
    prompt = ChatPromptTemplate.from_template("""
You are an expert chef assistant. 
Answer the user's question in a helpful and concise manner, 
based solely on the following context:
<context>
{context}
</context>
Question: {input}
#Rules:
- if the user wants a specific recipe, you must provide in the answer Ingredienti, Steps, Link. in this exact order.
                                              
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain