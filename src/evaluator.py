from langchain.evaluation import load_evaluator
from langchain_community.llms import huggingface_hub

'''EVAL MODEL'''

def get_accuracy_evaluator():

    judge_llm = huggingface_hub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 1024}
    )

    # eval model langChain.
    evaluator = load_evaluator("contextual_accuracy", llm=judge_llm)
    
    return evaluator