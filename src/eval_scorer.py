from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_percentage_scorer(model_name: str) -> PromptTemplate:
    """
    Crea una chain LLM che agisce come "meta-giudice".
    Legge il ragionamento di un primo giudice e assegna un punteggio percentuale.
    """

    scorer_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    prompt = PromptTemplate.from_template(
"""
You are a Score Assigner. Your sole task is to provide a percentage score based on a detailed analysis.
You will be given a user 'Question', an AI 'Answer', and a 'reasoning' from a primary judge.
The 'Judge's Reasoning' explains in detail why the 'Answer' is correct or not.

Based EXCLUSIVELY on the provided 'reasoning', assign a numerical score from 0 to 100 representing the accuracy of the 'answer'.
- If the reasoning notes minor errors or omissions, assign a score between 80 and 95.
- If the reasoning identifies significant inaccuracies, assign a score between 40 and 70.
- If the reasoning concludes that the answer is completely wrong (usually ending with "N"), assign a low score.

**You must respond with a single integer between 0 and 100. Do not add any text, explanations, or the '%' symbol.**

---
Question: {question}
Answer: {answer}
Judge's Reasoning: {reasoning}
---
Accuracy Score:
"""
    )

    return prompt | scorer_llm | StrOutputParser()