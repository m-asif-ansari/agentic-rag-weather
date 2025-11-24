# Agentic RAG & Weather Application

A production-ready Gen AI application that intelligently routes between weather API calls and PDF document queries using LangGraph, LangChain, and Qdrant.


## 🏗️ Architecture

```
User Query → LangGraph Agent → Intent Classification
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
            OpenWeather API Node        PDF RAG Node
                    ↓                       ↓
                    └───────────┬───────────┘
                                ↓
                    LLM Response Generation
                                ↓
                         Final Answer
```

## 🌟 Features

- **Intelligent Intent Classification**: Uses LLM to determine whether to fetch weather data or query PDF documents
- **Real-time Weather Data**: Fetches current weather information from OpenWeatherMap API
- **RAG System**: Retrieval-Augmented Generation for answering questions from PDF documents
- **Vector Database**: Uses Qdrant for efficient similarity search
- **LangGraph Workflow**: Agentic pipeline with conditional routing
- **LangSmith Integration**: Response evaluation and tracing
- **Comprehensive Tests**: Unit tests for all major components
- **Streamlit UI**: Interactive chat interface

## 📋 Prerequisites

- Python 3.10+
- Google Gemini API Key
- OpenWeatherMap API Key
- LangSmith API Key

## 🚀 Installation

### 1. Clone or create the project

```bash
mkdir agentic-rag-weather
cd agentic-rag-weather
git clone https://github.com/m-asif-ansari/agentic-rag-weather.git
```

### 2. Create virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENWEATHER_API_KEY=your-openweather-api-key-here
LANGSMITH_API_KEY=your-langsmith-api-key-here 
```

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Then open your browser to `http://localhost:8501`


## 🎯 Using the Application

### Weather Queries
- "What's the weather in New Delhi?"
- "Tell me about the temperature in Tokyo"
- "How's the weather in London today?"

### PDF Queries
1. Upload a PDF using the sidebar
2. Click "Index PDF" to process it
3. Ask questions like:
   - "Summarize the main points"
   - "What does the document say about [topic]?"
   - "Extract key information from the document"

## 🧪 Testing

The application includes comprehensive unit tests:

- **TestWeatherAPI**: Tests weather API integration
- **TestRAGSystem**: Tests RAG initialization and operations
- **TestAgenticPipeline**: Tests LangGraph workflow

### Running Tests

```bash
pytest -v
```



## 🏗️ Project Structure

```text
project-root/
│
├── db/
│   └── qdrant_db/                 # Local Qdrant vector database storage
│       └── collection/            # Qdrant collection data
│
├── streamlit_app/
│   │
│   ├── backend/                   # Core backend logic for the Streamlit app
│   │   ├── graph.py               # Agent graph implementation (LangGraph)
│   │   ├── llm_config.py          # LLM and Embeddings configuration
│   │   ├── open_weather.py        # Weather API integration
│   │   ├── rag_pipeline.py        # RAG pipeline and vector search logic
│   │   └── state.py               # Shared state models for agent workflow
│   │
│   ├── tests/                     # Unit tests for backend components
│   │   └── test_graph.py          # Tests for graph.py
│   │
│   └── app.py                     # Main Streamlit application entry point
│   
├── .env                           # Runtime environment variables
├── .env.example                   # Sample environment variable template
├── mini_reqs.txt                  # Minimal dependencies list
├── past_conversations.json        # Saved conversation logs
├── README.md                      # Project README file
└── requirements.txt               # Full Python dependencies
```
