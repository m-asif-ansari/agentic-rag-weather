import streamlit as st
from backend.rag_pipeline import load_and_index_pdf, empty_vector_store
from backend.graph import process_query



st.set_page_config(
    page_title="Agentic RAG & Weather App",
    initial_sidebar_state="expanded",
    page_icon="🤖",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/m-asif-ansari/",
        "About": "# This is a ChatBot. Made by- Asif Ansari",
    },
)

# Displaying the logo
st.logo(
    "https://mintlify.s3.us-west-1.amazonaws.com/agentai/logo/light.png", size="large"
)

st.title("🤖 Agentic RAG & Weather Application")
st.markdown("*Built by Langchain, LangGraph, Google Gemini, and Qdrant DB*")

# Display example queries
with st.expander("💡 Example Queries"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Weather Queries:**
        - What's the weather in New Delhi?
        - Tell me about the temperature in Tokyo
        - How's the weather in London today?
        
        
        """)
    with col2:
        st.markdown("""
        **PDF Queries:**
        - Summarize the main points of the document
        - What does the document say about [topic]?
        - Explain more about the [topic]?)
    """)
# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    if st.button(':warning: Reset Vector DB', type='primary'):
        st.toast(empty_vector_store())

    st.header("📄 PDF Upload")
    uploaded_file = st.file_uploader("Upload PDF for RAG", type=['pdf'], accept_multiple_files=True,)
    
    if uploaded_file:
        if st.button("Index PDFs"):
            with st.spinner("Processing PDFs..."):
                for file in uploaded_file:
                    # Save uploaded file temporarily
                    with open("temp.pdf", "wb") as f:
                        f.write(file.getbuffer())
                    
                    result = load_and_index_pdf("temp.pdf")
                    
                    if result["status"] == "success":
                        st.success(f"✅ Indexed file {file.name} into {result['chunks']} chunks!")
                    else:
                        st.error(f"❌ Error: while indexing file {file.name} - {result['message']}")

# Main chat interface
st.header("💬 Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.empty()

# Chat input
if prompt := st.chat_input("Ask about weather or your PDF document..."):        
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Process query
                response = process_query(prompt)
                st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})




