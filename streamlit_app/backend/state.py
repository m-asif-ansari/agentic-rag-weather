from typing import TypedDict, Annotated, Literal
from operator import add

class IntentState(TypedDict):
    """State for the Intent Classification"""
    intent: Literal["weather", "pdf"]
    city: str
    

class AgentState(TypedDict):
    """State for the Agentic Workflow"""
    user_query: str
    intent: str
    city: str
    weather_data: dict
    pdf_context: str
    final_response: str