import pytest
from unittest.mock import Mock, patch
from backend.graph import (
    classify_intent,
    fetch_weather,
    fetch_pdf_context,
    generate_response
)
from backend.state import AgentState


# Fixtures
@pytest.fixture
def base_state():
    """Base state for testing"""
    return AgentState(
        user_query="What's the weather in London?",
        intent="",
        city="",
        weather_data={},
        pdf_context="",
        final_response=""
    )


@pytest.fixture
def weather_state():
    """State with weather intent"""
    return AgentState(
        user_query="What's the weather in London?",
        intent="weather",
        city="London",
        weather_data={},
        pdf_context="",
        final_response=""
    )


@pytest.fixture
def pdf_state():
    """State with PDF intent"""
    return AgentState(
        user_query="What is mentioned in the document?",
        intent="pdf",
        city="",
        weather_data={},
        pdf_context="",
        final_response=""
    )



@patch('backend.graph.fetch_weather_details')
def test_weather_api_successful_response(mock_api, weather_state):
    """Test successful API response with complete weather data"""
    expected_data = {
        'city': 'London',
        'temperature': 15.5,
        'temp_min': 12.0,
        'temp_max': 18.0,
        'feels_like': 14.0,
        'humidity': 65,
        'description': 'Partly cloudy',
        'wind_speed': 3.5
    }
    mock_api.return_value = expected_data
    
    result = fetch_weather(weather_state)
    
    assert result['weather_data'] == expected_data
    assert result['weather_data']['city'] == 'London'
    assert result['weather_data']['temperature'] == 15.5
    mock_api.assert_called_once_with('London')


@patch('backend.graph.chat_llm')
def test_llm_weather_response_generation(mock_llm, weather_state):
    """Test LLM generates natural weather response"""
    weather_state['weather_data'] = {
        'city': 'London',
        'temperature': 15.5,
        'temp_min': 12.0,
        'temp_max': 18.0,
        'feels_like': 14.0,
        'humidity': 65,
        'description': 'Partly cloudy',
        'wind_speed': 3.5
    }
    mock_response = Mock()
    mock_response.content = "It's 15.5°C in London with partly cloudy skies."
    mock_llm.invoke.return_value = mock_response
    
    result = generate_response(weather_state)
    
    assert mock_llm.invoke.call_count == 1
    call_args = mock_llm.invoke.call_args[0][0]
    assert 'London' in call_args
    assert '15.5' in call_args
    assert 'Partly cloudy' in call_args
    assert result['final_response'] == "It's 15.5°C in London with partly cloudy skies."



@patch('backend.graph.retrieve_context')
def test_retrieval_successful(mock_retrieve, pdf_state):
    """Test successful context retrieval"""
    expected_context = "This document discusses artificial intelligence and its applications."
    mock_retrieve.return_value = expected_context
    
    result = fetch_pdf_context(pdf_state)
    
    assert result['pdf_context'] == expected_context
    mock_retrieve.assert_called_once_with(pdf_state['user_query'])