from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

'''RETRIVAL AUGMENTED GENERATION'''

def create_rag_agent(db, model_name: str) :

    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.0
    )

    '''--- RETRIEVER PROMPT ---'''
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    '''--- HISTORY_AWARE RETRIEVER ---'''
    retriever = db.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)

    '''--- PROMPT TEMEPLATE ---'''
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are BOB an expert culinary assistant. 
Answer the user's question in a helpful and concise manner, 
based solely on the following context which is a ricettario of recipes:
<context>
{context}
</context>
Question: {input}
Rules:
- when you give a specific recipe, you must provide in the answer: Ingredienti, Steps (as much as they are), Link. in this exact order.
- the answer must belong to the context.
- if chat_history is empty, you greatly welcome the user and introduce yourself.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return retrieval_chain