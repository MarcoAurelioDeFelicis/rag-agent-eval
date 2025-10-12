# FILE: main.py (CORRETTO)

import sys
import os
import argparse
import logging
import traceback
from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import HumanMessage, AIMessage

'''MAIN CONFIG'''
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import settings
from src.keys_config import configure_api_keys 
from src.vector_store import create_vector_store
from src.rag_agent import create_rag_agent
from src.evaluator import get_accuracy_evaluator
from src.eval_scorer import get_percentage_scorer

import signal
import sys

def signal_handler(sig, frame):
    print("\n⛔ Execution interrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout 
)

""" EVAL WORKFLOW """
def run_evaluation(user_input, response):
    logging.info("--- Starting Evaluation ---")

    if "context" not in response or not response["context"]:
        logging.warning("Evaluation skipped: no context was found in the response.")
        return

    context_str = "\n\n---\n\n".join(
        [doc.page_content for doc in response["context"]])
    
    # --- Judge's evaluation with fallback ---
    try:
        judge = get_accuracy_evaluator(model_name=settings.JUDGE_LLM_MODEL)
        eval_result = judge.evaluate_strings(
            prediction=response["answer"], input=user_input, reference=context_str
        )
    except ResourceExhausted:
        logging.warning(f"Falling back to '{settings.FALLBACK_LLM_MODEL}'.")
        judge = get_accuracy_evaluator(model_name=settings.FALLBACK_LLM_MODEL)
        eval_result = judge.evaluate_strings(
            prediction=response["answer"], input=user_input, reference=context_str
        )

    score_map = {1.0: "ACCURATE", 0.0: "NOT ACCURATE"}
    logging.info(f"Judge's Result: The answer is {score_map.get(eval_result.get('score'), 'UNKNOWN')}.")
    logging.info(f"Judge's Reasoning: {eval_result.get('reasoning')}")

    # --- Scorer evaluation with fallback ---
    logging.info("Calculating percentage score...")
    score_input = {
        "question": user_input,
        "answer": response["answer"], 
        "reasoning": eval_result.get('reasoning', '')
    }
    
    try:
        scorer = get_percentage_scorer(model_name=settings.SCORER_LLM_MODEL)
        raw_score_output = scorer.invoke(score_input)
    except ResourceExhausted:
        logging.warning(f"Quota exceeded for Scorer model '{settings.SCORER_LLM_MODEL}'. Falling back to '{settings.FALLBACK_LLM_MODEL}'.")
        scorer = get_percentage_scorer(model_name=settings.FALLBACK_LLM_MODEL)
        raw_score_output = scorer.invoke(score_input)

    try:
        cleaned_score_str = "".join(filter(str.isdigit, raw_score_output))
        percentage_score = int(cleaned_score_str)
        logging.info(f"Dynamic Accuracy Score: {percentage_score}%")
    except (ValueError, TypeError):
        logging.warning("Could not determine a percentage score from the model's output.")
    

""" RAG WORKFLOW """

def main(args):
    try:
        # --- INITIALIZATION ---
        configure_api_keys()
        db = create_vector_store(
            file_path=settings.CSV_FILE_PATH,
            persist_directory=settings.DB_PERSIST_DIRECTORY
        )
        primary_rag_chain = create_rag_agent(db, model_name=settings.RAG_LLM_MODEL)
        fallback_rag_chain = create_rag_agent(db, model_name=settings.FALLBACK_LLM_MODEL)
        logging.info("🧠 RAG Culinary Assistant is ready!")

        # --- CHAT LOOP ---
        last_user_input, last_response = None, None
        chat_history = [] 
        
        print("\nTO START: Write your questions about the recipes (or 'quit' to close the chat).")

        while True:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() == 'quit':
                print("👋 See you next time!")
                break
            
            if user_input.lower() == '/eval':
                if last_user_input and last_response:
                    try:
                        run_evaluation(last_user_input, last_response)
                    except Exception as e:
                        logging.error(f"Evaluation failed with an error: {e}")
                        print("\n🤖 Assistant: Evaluation failed. Check logs for details.")
                else:
                    print("\n🤖 Assistant: You must ask a question before you can evaluate an answer.")
                continue

            try:
                invoke_payload = {"input": user_input, "chat_history": chat_history}
                response = primary_rag_chain.invoke(invoke_payload)
            except ResourceExhausted:
                logging.warning(f"Quota exceeded for RAG model '{settings.RAG_LLM_MODEL}'. Falling back to '{settings.FALLBACK_LLM_MODEL}'.")
                invoke_payload = {"input": user_input, "chat_history": chat_history}
                response = fallback_rag_chain.invoke(invoke_payload)

            

            # --- DEBUGGING: STAMPIAMO IL CONTESTO ---
            print("\n" + "="*50)
            print("🔍 CONTEXT RETRIEVED AND PASSED TO LLM:")
            if "context" in response and response["context"]:
                for i, doc in enumerate(response["context"]):
                    print(f"--- Document {i+1} ---\n{doc.page_content}\n")
            else:
                print("!!! NO CONTEXT RETRIEVED !!!")
            print("="*50 + "\n")
            # --- FINE DEBUGGING ---
            
            print("\n🤖 Assistant:", response.get("answer", "Sorry, I couldn't generate a response."))
            
            # --- AGGIORNAMENTO DELLA CRONOLOGIA E DELLO STATO ---
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response.get("answer", "")))

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            last_user_input, last_response = user_input, response

            # --- ESECUZIONE DELLA VALUTAZIONE AUTOMATICA ---
            if args.evaluate:
                try:
                    run_evaluation(user_input, response)
                except ResourceExhausted:
                    logging.error("API quota exceeded for BOTH primary and fallback models during automatic evaluation.")
                    print("\n🤖 Assistant: Automatic evaluation failed due to API usage limits.")

    except Exception:
        logging.error("An unexpected error occurred. Printing full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Culinary Assistant")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable automatic evaluation of the RAG agent's answers after each response."
    )
    args = parser.parse_args()
    main(args)