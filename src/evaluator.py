from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI


'''EVAL MODEL'''

def get_accuracy_evaluator(model_name: str):

    judge_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        
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