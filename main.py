
from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# Initialize the LLM used for ticket analysis and responses
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


# Shared state passed between LangGraph nodes
class TicketState(TypedDict, total=False):
    ticket: str
    category: str
    severity: str
    sufficient_info: bool
    in_scope: bool
    action: str
    response: str


# Structured output ensures predictable LLM classifications
class TicketAnalysis(BaseModel):
    category: Literal[
        "Authentication", "Kubernetes", "Terraform",
        "Performance", "LangSmith", "Other"
    ]
    severity: Literal["low", "medium", "high", "critical"]
    sufficient_info: bool
    in_scope: bool


# Node 1: Analyze ticket using the LLM
def analyze_ticket(state: TicketState):
    prompt = f"""
    You are a technical support engineer.

    Analyze the following customer support ticket:
    {state['ticket']}

    Identify:
    1. Issue category
    2. Severity based on customer impact
    3. Whether enough information exists for initial troubleshooting
    4. Whether the request is related to technical support

    Severity guidelines:
    - critical: Widespread production outage or major business disruption.
    - high: Major functionality unavailable or significant degradation.
    - medium: Limited or intermittent customer impact.
    - low: Minor issues, informational requests, or no demonstrated impact.

    Category guidelines:
    - Classify based only on technical evidence explicitly
    mentioned in the customer's ticket.
    - Do not infer Authentication simply because users
    cannot access an application.
    - If the underlying technical issue is not identifiable,
    use the category "Other".
    - Severity and category are independent:
    a critical outage can still have category "Other".

    Set in_scope to false for unrelated questions such as
    geography, general knowledge, or entertainment.

    Do not assume customer impact or root causes not provided.
    """

    analysis = llm.with_structured_output(
        TicketAnalysis
    ).invoke(prompt)

    return {
        "category": analysis.category,
        "severity": analysis.severity,
        "sufficient_info": analysis.sufficient_info,
        "in_scope": analysis.in_scope
    }


# Conditional routing: Python decides which node executes next
def route_ticket(state: TicketState):
    if not state["in_scope"]:
        return "clarify"
    elif state["severity"] == "critical":
        return "escalate"
    elif not state["sufficient_info"]:
        return "clarify"
    else:
        return "troubleshoot"


# Node 2a: Ask questions or redirect unrelated requests
def clarify(state: TicketState):
    if not state["in_scope"]:
        return {
            "action": "clarify",
            "response": (
                "I can help with technical support issues. "
                "Could you share the technical problem you're experiencing?"
            )
        }

    response = llm.invoke(
        f"""You are a technical support engineer.

        Customer ticket: {state['ticket']}

        Ask up to three relevant clarifying questions.
        Do not guess the root cause or request secrets."""
    )

    return {
        "action": "clarify",
        "response": response.content
    }


# Node 2b: Provide safe initial diagnostic steps
def troubleshoot(state: TicketState):
    response = llm.invoke(
        f"""You are a technical support engineer.

        Customer ticket: {state['ticket']}
        Category: {state['category']}

        Suggest the 3 most relevant initial diagnostic steps.
        Prioritize non-destructive checks.
        Do not assume an unverified root cause.
        Do not recommend disruptive production changes
        without appropriate impact assessment and approval."""
    )

    return {
        "action": "troubleshoot",
        "response": response.content
    }


# Node 2c: Recommend escalation for critical incidents
def escalate(state: TicketState):
    response = llm.invoke(
        f"""You are a technical support engineer.

        Critical customer issue: {state['ticket']}

        Acknowledge the business impact.
        Recommend immediate escalation to the appropriate teams.
        Suggest up to 3 safe initial diagnostic steps.
        Do not claim escalation has already happened.
        Do not assume an unverified root cause."""
    )

    return {
        "action": "escalate",
        "response": response.content
    }


# Build the LangGraph workflow and connect its nodes
builder = StateGraph(TicketState)

builder.add_node("analyze_ticket", analyze_ticket)
builder.add_node("clarify", clarify)
builder.add_node("troubleshoot", troubleshoot)
builder.add_node("escalate", escalate)

builder.add_edge(START, "analyze_ticket")

# Route tickets based on severity, scope, and available information
builder.add_conditional_edges(
    "analyze_ticket",
    route_ticket,
    {
        "clarify": "clarify",
        "troubleshoot": "troubleshoot",
        "escalate": "escalate"
    }
)

builder.add_edge("clarify", END)
builder.add_edge("troubleshoot", END)
builder.add_edge("escalate", END)

graph = builder.compile()


# Execute the workflow from the terminal
if __name__ == "__main__":
    ticket = input("Enter customer support ticket: ")
    result = graph.invoke({"ticket": ticket})

    print("\n--- TRIAGE RESULT ---")
    print("Category:", result["category"])
    print("Severity:", result["severity"])
    print("Action:", result["action"])
    print("\nResponse:")
    print(result["response"])
