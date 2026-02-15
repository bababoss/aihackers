from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()  # Add this line
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_ENDPOINT")     # Custom endpoint
)

# llm = ChatOpenAI(
#     model="ollama/mistral",
#     base_url="http://localhost:8000/v1",
#     api_key="any-key"
# )

# ---- STATE ----
class AgentState(TypedDict):
    question: str
    task_type: Literal["explain", "calculate"]
    answer: str

def router_node(state: AgentState):
    q = state["question"]

    prompt = f"""
Classify this STEM query:

Question: {q}

Return only one word:
- explain
- calculate
"""

    result = llm.invoke(prompt).content.strip().lower()

    return {
        "task_type": result
    }

def explanation_node(state: AgentState):
    q = state["question"]

    response = llm.invoke(
        f"Explain clearly for a student:\n{q}"
    )

    return {
        "answer": response.content
    }
def calculation_node(state: AgentState):
    q = state["question"]

    response = llm.invoke(
        f"Solve this numerically step by step:\n{q}"
    )

    return {
        "answer": response.content
    }
if __name__ == "__main__":
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("explain", explanation_node)
    graph.add_node("calculate", calculation_node)

    # Entry
    graph.set_entry_point("router")

    # Conditional routing
    graph.add_conditional_edges(
        "router",
        lambda state: state["task_type"],
        {
            "explain": "explain",
            "calculate": "calculate",
        },
    )

    # End nodes
    graph.add_edge("explain", END)
    graph.add_edge("calculate", END)

    stem_agent = graph.compile()
    result = stem_agent.invoke(
        {
            "question": "Why does increasing temperature increase resistance in metals?"
        }
    )

    print(result["answer"])

