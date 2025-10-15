import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.settings import DEFAULT_RETRIEVER_TOP_K

def create_dynamic_retriever_chain(db: VectorStore, llm: ChatGoogleGenerativeAI, top_n: int) -> Runnable:
 
    # --- RETRIEVERS ---
    if top_n >= DEFAULT_RETRIEVER_TOP_K:
        top_k = top_n * 2
        base_retriever = db.as_retriever(search_kwargs={"k": top_k })
    else:
        base_retriever = db.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVER_TOP_K})
    
    logging.info(f"### Base retriever initialized with top_k={top_k}, dynamic top_n={top_n}.")

    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    # --- RE-RANKER + DYNAMIC TOP N TOP K---
    reranker = FlashrankRerank(top_n=top_n)
    
    # --- COMPRESSION RETRIEVER (wrapper) ---
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=multi_query_retriever
    )

    # --- RETRIEVER HISTORY AWARE (wrapper) ---
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Retrieve the most relevant documents by also taking the chat history into account."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=compression_retriever, 
        prompt=retriever_prompt
    )

    return history_aware_retriever