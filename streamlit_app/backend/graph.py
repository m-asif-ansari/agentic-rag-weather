from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
from backend.state import AgentState, IntentState
from backend.llm_config import chat_llm, structured_llm
from backend.open_weather import fetch_weather_details
from backend.rag_pipeline import retrieve_context



def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent using LLM"""

    prompt = f"""
        Classify the following user query into one of these categories:
        - weather: Questions about weather, temperature, climate conditions
        - pdf: Questions about document content, information retrieval and non weather related

        And extract the city name if the intent is weather related
        - city : Extract the city name 
        
        User query: {state['user_query']}
    """

    response = structured_llm.invoke(prompt)
    intent = response['intent']
    city = response['city']

    state['intent'] = intent if intent in ['weather', 'pdf'] else 'pdf'
    state['city'] =  city if city else " "
    
    return state

def route_query(state: AgentState) -> Literal["weather", "pdf"]:
    """Route based on classified intent"""

    return state['intent']

def fetch_weather(state: AgentState) -> AgentState:
    """Fetch weather data"""
    
    weather_data = fetch_weather_details(state['city'])
    state['weather_data'] = weather_data
    
    return state

def fetch_pdf_context(state: AgentState) -> AgentState:
    """Fetch relevant PDF context"""
    
    context = retrieve_context(state['user_query'])
    state['pdf_context'] = context
    
    return state

def generate_response( state: AgentState) -> AgentState:
    """Generate final response using LLM"""
    
    if state['intent'] == 'weather':
        weather = state.get('weather_data', {})
        if 'error' in weather:
            response = f"I'm sorry, I couldn't fetch the weather data. {weather['error']}"
        else:
            prompt = f"""
                Based on this weather data, provide a natural, conversational response:
                
                City: {weather.get('city', 'Unknown')}
                Temperature: {weather.get('temperature', 'N/A')}°C
                Min_Temp: {weather.get('temp_min', 'N/A')}°C
                Max_Temp: {weather.get('temp_max', 'N/A')}°C
                Feels Like: {weather.get('feels_like', 'N/A')}°C
                Humidity: {weather.get('humidity', 'N/A')}%
                Description: {weather.get('description', 'N/A')}
                Wind Speed: {weather.get('wind_speed', 'N/A')} m/s
                
                User query: {state['user_query']}
            """
            response = chat_llm.invoke(prompt).content
    else:
        context = state.get('pdf_context', 'No context available')
        prompt = f"""
            You are a helpful, knowledgeable AI assistant. Your job is to answer the user's question accurately and clearly.

            Instructions:
            - Read the user's question: {state['user_query']}

            - Review the additional context provided:
                {context}

            - Use the context only if it is relevant to the user's question.
                - If the context contains useful information, incorporate it into your answer.
                - If the context is not relevant, ignore it and answer the question normally.

            - If the context does not contain any information related to the question, politely mention that the context was not relevant but still provide the best possible answer.
        """
        response = chat_llm.invoke(prompt).content
    
    state['final_response'] = response
    
    return state




def build_graph() -> StateGraph:
    """Build LangGraph workflow"""

    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("fetch_weather", fetch_weather)
    workflow.add_node("fetch_pdf_context", fetch_pdf_context)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.set_entry_point("classify_intent")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_query,
        {
            "weather": "fetch_weather",
            "pdf": "fetch_pdf_context"
        }
    )
    
    workflow.add_edge("fetch_weather", "generate_response")
    workflow.add_edge("fetch_pdf_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()


def process_query(query: str) -> str:
    """Process user query through the graph"""

    initial_state = AgentState(
        user_query=query,
        intent="",
        weather_data={},
        pdf_context="",
        final_response="",
    )
    graph = build_graph()
    result = graph.invoke(initial_state)

    # saving all data in file for logging purpose
    with open("past_conversations.json", "a") as json_file:
        json.dump(result, json_file, indent=4) 

    return result['final_response']