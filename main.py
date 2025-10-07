import sys
import os
import traceback 

'''MAIN CONFIG'''
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#added this for the tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config import configure_api_keys
from src.vector_store import create_vector_store
from src.rag_agent import create_rag_agent
from src.evaluator import get_accuracy_evaluator
from src.eval_scorer import get_percentage_scorer

""" WORKFLOW"""

def main():

    try:
        """ CONFIGS """
        configure_api_keys()

        csv_file_path = "data/gz_recipe.csv"
        
        db = create_vector_store(csv_file_path)

        rag_chain = create_rag_agent(db)
        print("\n🧠  RAG Culinary Assistant's ready!")
        
        accuracy_evaluator = get_accuracy_evaluator()
        percentage_scorer = get_percentage_scorer()
        print("⚖️  Evaluators (Judge & Scorer) are ready!")


        """ CHAT """

        print("\n TO START: Write your questions about the recipes (or 'quit' to close the chat).") 
        while True:
            user_input = input("\n👤 You: ")
            if user_input.lower() == 'quit':
                print("👋 See you next time!")
                break
            
            response = rag_chain.invoke({"input": user_input})
            print("\n🤖 Assistant:", response["answer"])

            """ EVALUATION (judge)"""

            print("\n--- Evaluation ---")

            context_str = "\n\n---\n\n".join(
                [doc.page_content for doc in response["context"]])

            eval_result = accuracy_evaluator.evaluate_strings(
                prediction=response["answer"],      
                input=user_input,                   
                reference=context_str               
            )

            score_map = {1.0: "ACCURATE", 0.0: "NOT ACCURATE"}
            print(f"Result: The answer is {score_map.get(eval_result['score'], 'UNKNOWN')}.")
            print(f"Judge's Reasoning: {eval_result['reasoning']}")
            print("------------------")
            
            """ EVALUATION (score)"""

            print("\n📊 Calculating percentage score...")
            score_input = {
                "question": user_input,
                "answer": response["answer"],
                "reasoning": eval_result['reasoning']
                }
            
            raw_score_output = percentage_scorer.invoke(score_input)

            try:
                cleaned_score_str = "".join(filter(str.isdigit, raw_score_output))
                percentage_score = int(cleaned_score_str)
            except (ValueError, TypeError):
                print("⚠️ Could not determine a percentage score.")
                percentage_score = None

            if percentage_score is not None:
             print(f"Dynamic Accuracy Score: {percentage_score}%")
            



    except Exception as e:
        # print(f"\n STOP \n ❌ ERROR: {e}")
        print(f"\n STOP \n ❌ An error occurred. Printing full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()