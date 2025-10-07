from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI

'''EVAL MODEL'''

def get_accuracy_evaluator():

    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.1,
        
        model_kwargs={"safety_settings": {
            "HARASSMENT": "BLOCK_NONE",
            "HATE": "BLOCK_NONE",
            "SEXUAL": "BLOCK_NONE",
            "DANGEROUS": "BLOCK_NONE",
        }}
    )

    evaluator = load_evaluator(
        "labeled_criteria",
        criteria="correctness",
        llm=judge_llm
    )


    return evaluator