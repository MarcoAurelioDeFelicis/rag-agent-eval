import sys
import os
import argparse
import logging
import traceback
from google.api_core.exceptions import ResourceExhausted
import signal

'''MAIN CONFIG'''
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import settings
from src.keys_config import configure_api_keys
from src.vector_store import create_vector_store
from src.rag_orchestrator import RAGorchestrator
from src.eval.evaluator import get_accuracy_evaluator
from src.eval.eval_scorer import get_percentage_scorer

# quit (Ctrl+C) TODO: fix it
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

def run_evaluation(user_input, answer, context):
    logging.info("--- Starting Evaluation ---")

    if not context:
        logging.warning("Evaluation skipped: no context was retrieved.")
        print("\n🤖 Assistant: Evaluation skipped, no relevant context was found for the last question.")
        return

    context_str = "\n\n---\n\n".join([doc.page_content for doc in context])
    
    try:
        judge = get_accuracy_evaluator(model_name=settings.JUDGE_LLM_MODEL)
        eval_result = judge.evaluate_strings(prediction=answer, input=user_input, reference=context_str)
    except ResourceExhausted:
        logging.warning(f"Quota exceeded for Judge model. Falling back to '{settings.FALLBACK_LLM_MODEL}'.")
        judge = get_accuracy_evaluator(model_name=settings.FALLBACK_LLM_MODEL)
        eval_result = judge.evaluate_strings(prediction=answer, input=user_input, reference=context_str)

    score_map = {1.0: "ACCURATE", 0.0: "NOT ACCURATE"}
    logging.info(f"Judge's Result: The answer is {score_map.get(eval_result.get('score'), 'UNKNOWN')}.")
    logging.info(f"Judge's Reasoning: {eval_result.get('reasoning')}")

    score_input = {"question": user_input, "answer": answer, "reasoning": eval_result.get('reasoning', '')}
    
    try:
        scorer = get_percentage_scorer(model_name=settings.SCORER_LLM_MODEL)
        raw_score_output = scorer.invoke(score_input)
    except ResourceExhausted:
        logging.warning(f"Quota exceeded for Scorer. Falling back to '{settings.FALLBACK_LLM_MODEL}'.")
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
        configure_api_keys()
        db = create_vector_store(
            file_path=settings.CSV_FILE_PATH,
            persist_directory=settings.DB_PERSIST_DIRECTORY
        )
        orchestrator = RAGorchestrator(db)
        logging.info("🧠 RAG Culinary Assistant is ready!")

        # --- CHAT LOOP ---
        last_user_input, last_answer = None, None
        
        print("\nTO START: Write your questions about the recipes (or 'quit' to close the chat).")

        while True:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() == 'quit':
                print("👋 See you next time!")
                break
            
            if user_input.lower() == '/eval':
                if last_user_input and last_answer:
                    try:
                        run_evaluation(last_user_input, last_answer, orchestrator.last_retrieved_docs)
                    except Exception as e:
                        logging.error(f"Evaluation failed with an error: {e}", exc_info=True)
                        print("\n🤖 Assistant: Evaluation failed. Check logs for details.")
                else:
                    print("\n🤖 Assistant: You must ask a question before you can evaluate an answer.")
                continue

            #--- INVOKE RAG ORCHESTRATOR ---
            answer = orchestrator.invoke(user_input)
            
            #--- CONTEXT RETRIEVING ---
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
            
            last_user_input, last_answer = user_input, answer

            # --- EVALUATION ---
            if args.evaluate:
                try:
                    run_evaluation(user_input, answer, orchestrator.last_retrieved_docs)
                except ResourceExhausted:
                    logging.error("API quota exceeded for BOTH models during auto-evaluation.")
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