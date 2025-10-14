import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, AIMessage
from google.api_core.exceptions import ResourceExhausted


from src.rag_flow.query_analyzer import create_query_analyzer_chain
from src.rag_flow.retriver import create_dynamic_retriever_chain
from src.rag_flow.qa_chain import create_qa_chain
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
        
        self.chat_history = []
        self.last_retrieved_docs = []

    def invoke(self, user_input: str) -> str:
        # --- GET DYNAMIC TOP N ---
        try:
            analysis_result = self.query_analyzer.invoke({"user_query": user_input})
            top_n = int(analysis_result)
            logging.info(f"Dynamic top_n detected: {top_n}")
        except (ValueError, TypeError):
            top_n = DEFAULT_RERANK_TOP_N
            logging.info(f"No specific number detected. Using default top_n: {top_n}")

        try:
            retriever_chain = create_dynamic_retriever_chain(self.db, self.primary_llm, top_n)
            self.last_retrieved_docs = retriever_chain.invoke({
                "input": user_input, "chat_history": self.chat_history
            })
            
            response = self.qa_chain.invoke({
                "input": user_input, "chat_history": self.chat_history, "context": self.last_retrieved_docs
            })
        except ResourceExhausted as e:
            logging.warning(f"Primary LLM failed: {e}. Falling back to '{FALLBACK_LLM_MODEL}'.")

            # --- FALLBACK LLM ---
            retriever_chain = create_dynamic_retriever_chain(self.db, self.fallback_llm, top_n)
            self.last_retrieved_docs = retriever_chain.invoke({
                "input": user_input, "chat_history": self.chat_history
            })
            response = self.fallback_qa_chain.invoke({
                "input": user_input, "chat_history": self.chat_history, "context": self.last_retrieved_docs
            })
        
        # --- UPDATE CHAT HISTORY ---
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        if len(self.chat_history) > 20: 
            self.chat_history = self.chat_history[-20:] 
        
        return response