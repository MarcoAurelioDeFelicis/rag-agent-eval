from typing import TypedDict, Any

class AppState(TypedDict):

    rag_orchestrator: Any  # Per evitare import circolari, usiamo Any
    user_input: str
    last_question: str
    last_answer: str
    auto_evaluate: bool
    should_exit: bool