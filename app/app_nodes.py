import logging
from src.app_state import AppState

# La funzione di valutazione rimane in main.py, la passiamo come argomento
# per mantenere i nodi focalizzati sulla logica del grafo.

class AppNodes:

    def __init__(self, run_evaluation_func):
        self.run_evaluation = run_evaluation_func

    def process_question(self, state: AppState) -> AppState:

        print("\n🤖 Assistant: Thinking...")
        orchestrator = state["rag_orchestrator"]
        user_input = state["user_input"]
        answer = orchestrator.invoke(user_input)

        # Stampa il contesto recuperato
        print("\n" + "="*50)
        print("🔍 CONTEXT RETRIEVED AND PASSED TO LLM:")
        retrieved_context = orchestrator.last_retrieved_docs
        if retrieved_context:
            for i, doc in enumerate(retrieved_context):
                print(f"--- Document {i+1} ---\n{doc.page_content}\n")
        else:
            print("!!! NO CONTEXT RETRIEVED !!!")
        print("="*50 + "\n")

        print("\n🤖 Assistant:", answer)

        # Auto-evaluation se abilitata
        if state["auto_evaluate"]:
            try:
                self.run_evaluation(user_input, answer, retrieved_context)
            except Exception as e:
                logging.error(f"Auto-evaluation failed with an error: {e}", exc_info=True)
                print("\n🤖 Assistant: Automatic evaluation failed. Check logs for details.")
        
        return {
            "last_question": user_input,
            "last_answer": answer
        }

    def handle_evaluation(self, state: AppState) -> AppState:

        last_question = state.get("last_question")
        last_answer = state.get("last_answer")
        orchestrator = state["rag_orchestrator"]

        if last_question and last_answer:
            try:
                self.run_evaluation(last_question, last_answer, orchestrator.last_retrieved_docs)
            except Exception as e:
                logging.error(f"Evaluation failed with an error: {e}", exc_info=True)
                print("\n🤖 Assistant: Evaluation failed. Check logs for details.")
        else:
            print("\n🤖 Assistant: You must ask a question before you can evaluate an answer.")
        
        return {} # Non modifica lo stato principale

    def handle_exit(self, state: AppState) -> AppState:

        print("👋 See you next time!")
        return {"should_exit": True}

    def router(self, state: AppState) -> str:

        user_input = state["user_input"].lower()
        if user_input == 'quit':
            return "handle_exit"
        elif user_input == '/eval':
            return "handle_evaluation"
        else:
            return "process_question"