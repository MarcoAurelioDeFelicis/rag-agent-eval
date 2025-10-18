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
from src.app_graph import create_app_graph # NUOVO IMPORT

def signal_handler(sig, frame):
    print("\n⛔ Execution interrupted by user. Exiting...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout 
)

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

""" MAIN WORKFLOW """
def main(args):
    try:
        configure_api_keys()
        db = create_vector_store(
            file_path=settings.CSV_FILE_PATH,
            persist_directory=settings.DB_PERSIST_DIRECTORY
        )
        orchestrator = RAGorchestrator(db)
        app_graph = create_app_graph(run_evaluation_func=run_evaluation)

        logging.info("🧠 RAG Culinary Assistant is ready!")
        print("\nTO START: Write your questions about the recipes (or 'quit' to close the chat).")

        #--- INITIAL APP STATE ---
        app_state = {
            "rag_orchestrator": orchestrator,
            "last_question": "",
            "last_answer": "",
            "auto_evaluate": args.evaluate,
            "should_exit": False,
        }

        while not app_state["should_exit"]:
            user_input = input("\n👤 You: ")
            app_state["user_input"] = user_input
            result_state = app_graph.invoke(app_state)
            app_state.update(result_state)

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