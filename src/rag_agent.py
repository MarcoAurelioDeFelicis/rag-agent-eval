from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import Runnable


def create_rag_agent(db: VectorStore, model_name: str) -> Runnable:

    # --- LLM ---
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7
    )

    # --- RETRIEVER MULTI-QUERY ---
    base_retriever = db.as_retriever()
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    # --- RETRIVER PROMPT + CHAT HISTORY ---
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Retrieve the most relevant documents by also taking the chat history into account."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=multi_query_retriever,
        prompt=retriever_prompt
    )

    # --- ANSWER PROMPT ---
    prompt_answer = ChatPromptTemplate.from_messages([
        ("system", """
** Context **
You will receive a context containing culinary information (recipes, ingredients, or cooking notes) that must be used as the sole knowledge source for your answer.

** Role **
You are BOB, an expert chef assistant.

** Instruction **
Answer the user's question in a helpful and concise manner, based solely on the following context.

** Steps **
1. Read and understand the given context between <context> tags.
2. Process the user's question {input}.
3. Generate an answer that follows these rules:
    - You must answer and translate the context in the same language as the user question.
    - If the user wants a specific recipe, provide: Title, Ingredients, Category, Steps, Link (in this exact order).
    - Don't say "Based on the context"; call the context as "the recipe", "the recipes book", or similar.
    - if you are just greeting, or saying goodbye, or similar, do it in a friendly way, but do not mention the context if not necessay.
4. Exceptions:
    - ONLY if the User Explicitly ask you to "generate, create, invent" recipes, or similar, you can eventually use your prior knowledge of cooking, and take insipration from the contex, but the context is the main source.
         
** Parameters **
<context>
{context}
</context>
Question: {input}

** End goal **
Provide a precise, language-consistent culinary response using only the given context in the same lenguage of the user question.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # --- rag chain ---
    document_chain = create_stuff_documents_chain(llm, prompt_answer)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return retrieval_chain
