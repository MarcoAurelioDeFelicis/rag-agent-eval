import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#added this for the tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config import configure_api_key
from src.vector_store import create_vector_store
from src.rag_agent import create_rag_agent

def main():

    """ WORKFLOW"""
    try:
        configure_api_key()

        csv_file_path = "data/gz_recipe.csv"
        
        db = create_vector_store(csv_file_path)

        chain = create_rag_agent(db)

        """ CHAT """

        print("\n🤖 Culinary Assistant's ready!")
        print("Write your questions about the recipes (or 'quit' to close the chat).")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                print("👋 See you next time!")
                break
            
            """ INVOKE """
            response = chain.invoke({"input": user_input})
            
            print("\nAssistant:", response["answer"])

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()