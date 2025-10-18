from langgraph.graph import StateGraph, END
from app.app_state import AppState
from app.app_nodes import AppNodes

def create_app_graph(run_evaluation_func):

    nodes = AppNodes(run_evaluation_func)
    workflow = StateGraph(AppState)

    # --- NODES ---
    workflow.add_node("process_question", nodes.process_question)
    workflow.add_node("handle_evaluation", nodes.handle_evaluation)
    workflow.add_node("handle_exit", nodes.handle_exit)

    workflow.set_conditional_entry_point(
        nodes.router,
        {
            "process_question": "process_question",
            "handle_evaluation": "handle_evaluation",
            "handle_exit": "handle_exit",
        }
    )

    # Dopo ogni azione, il turno finisce e si torna al loop principale in main.py
    workflow.add_edge("process_question", END)
    workflow.add_edge("handle_evaluation", END)
    workflow.add_edge("handle_exit", END)

    app_graph = workflow.compile()
    return app_graph