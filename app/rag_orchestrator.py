import logging
import time  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, AIMessage
from google.api_core.exceptions import ResourceExhausted

from src.rag.query_analyzer import create_query_analyzer_chain
from src.rag.retriver import create_dynamic_retriever_chain
from src.rag.qa_chain import create_qa_chain
from src.settings import RAG_LLM_MODEL, FALLBACK_LLM_MODEL, DEFAULT_RERANK_TOP_N, ANALYZER_LLM_MODEL

class RAGorchestrator:
    def __init__(self, db: VectorStore):

        # --- LLM MODELS ---
        self.primary_llm = ChatGoogleGenerativeAI(model=RAG_LLM_MODEL, temperature=0.7)
        self.fallback_llm = ChatGoogleGenerativeAI(model=FALLBACK_LLM_MODEL, temperature=0.7)
        
        # --- STATIC COMPONENTS ---
        self.db = db
        self.query_analyzer = create_query_analyzer_chain(model_name=ANALYZER_LLM_MODEL)
        self.qa_chain = create_qa_chain(self.primary_llm)
        self.fallback_qa_chain = create_qa_chain(self.fallback_llm)
        
        # --- CHAT HISTORY & TIMEOUT MANAGEMENT ---
        self.chat_history = []
        self.last_retrieved_docs = []
        self.last_interaction_time = None  # Tiene traccia dell'ora dell'ultimo messaggio
        self.inactivity_timeout = 86400  # Timeout 1 day in seconds
    def invoke(self, user_input: str) -> str:
        
        # --- CHAT HISTORY E TIMEOUT HANDLING ---
        if self.last_interaction_time and (time.time() - self.last_interaction_time) > self.inactivity_timeout:
            logging.info(f"Timeout ({self.inactivity_timeout}s). Chat History Has Been Delete.")
            self.chat_history = []
        
        self.last_interaction_time = time.time()
        is_first_message = not self.chat_history
        

        # --- GET DYNAMIC TOP N ---
        try:
            analysis_result = self.query_analyzer.invoke({"user_query": user_input})
            top_n = int(analysis_result) + 3
            logging.info(f"Dynamic top_n detected: {top_n}")
        except (ValueError, TypeError):
            top_n = DEFAULT_RERANK_TOP_N
            logging.info(f"No specific number detected. Using default top_n: {top_n}")

        if is_first_message:
            qa_input_text = (
                f"Tis is the first interaction with the user. "
                f"Welcome them and introduce yourself as BOB, their culinary assistant, then you can reply to their question. "
                f"their question is: '{user_input}'"
            )
        else:
            qa_input_text = user_input

        try:
            retriever_chain = create_dynamic_retriever_chain(self.db, self.primary_llm, top_n)
            self.last_retrieved_docs = retriever_chain.invoke({
                "input": user_input, "chat_history": self.chat_history
            })
            
            response = self.qa_chain.invoke({
                "input": qa_input_text,  # Usa il testo modificato per il QA
                "chat_history": self.chat_history, 
                "context": self.last_retrieved_docs
            })
        except ResourceExhausted as e:
            logging.warning(f"Primary LLM failed: {e}. Falling back to '{FALLBACK_LLM_MODEL}'.")

            # --- FALLBACK LLM ---
            retriever_chain = create_dynamic_retriever_chain(self.db, self.fallback_llm, top_n)
            self.last_retrieved_docs = retriever_chain.invoke({
                "input": user_input, "chat_history": self.chat_history
            })
            response = self.fallback_qa_chain.invoke({
                "input": qa_input_text,  
                "chat_history": self.chat_history, 
                "context": self.last_retrieved_docs
            })
        
        # --- UPDATE CHAT HISTORY ---
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        if len(self.chat_history) > 20: 
            self.chat_history = self.chat_history[-20:] 
        
        return response