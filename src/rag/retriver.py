import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.prompts import PromptTemplate

from src.settings import DEFAULT_RETRIEVER_TOP_K

def create_dynamic_retriever_chain(db: VectorStore, llm: ChatGoogleGenerativeAI, top_n: int) -> Runnable:
 
    # --- RETRIEVERS ---
    if top_n >= DEFAULT_RETRIEVER_TOP_K:
        top_k = top_n * 2
    else:
        top_k = DEFAULT_RETRIEVER_TOP_K
    base_retriever = db.as_retriever(search_kwargs={"k": top_k})    
    logging.info(f"### Base retriever initialized with top_k={top_k}, dynamic top_n={top_n}.")

    # --- MULTI-QUERY RETRIEVER ---
    prompt_multiquery = PromptTemplate.from_template(
    """You are an AI language model assistant. Your task is
    to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector  database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions in the same language of the Original question,
      separated by newlines,. Original question: {question}""",)
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm, prompt=prompt_multiquery)

    # --- RE-RANKER + DYNAMIC TOP N TOP K---
    reranker = FlashrankRerank(top_n=top_n)
    
    # --- COMPRESSION RETRIEVER (wrapper) ---
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=multi_query_retriever
    )

    # --- RETRIEVER HISTORY AWARE (wrapper) ---
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Retrieve the most relevant documents by also taking the chat history into account."
        "The user often refers to something from the chat history, so use it to better understand the context and what to retrive."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=compression_retriever, 
        prompt=retriever_prompt
    )

    return history_aware_retriever