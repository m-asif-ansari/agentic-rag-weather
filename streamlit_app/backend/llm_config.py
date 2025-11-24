from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.state import IntentState

chat_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

structured_llm = chat_llm.with_structured_output(
    schema=IntentState, method="json_schema"
)

embeddings_llm = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")