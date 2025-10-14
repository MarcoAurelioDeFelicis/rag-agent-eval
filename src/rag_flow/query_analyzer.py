from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

def create_query_analyzer_chain(model_name: str) -> Runnable:
  
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    prompt = PromptTemplate.from_template(
        """DGiven the user's input, extract the number of items they are requesting.
- If the user asks for a specific number of results (e.g., "give me 5 recipes", "list 3 options"), respond ONLY with the number (e.g., "5", "3").
- If the user does not specify a number, respond ONLY with the word 'default'.

User input:
{user_query}
"""
    )

    return prompt | llm | StrOutputParser()