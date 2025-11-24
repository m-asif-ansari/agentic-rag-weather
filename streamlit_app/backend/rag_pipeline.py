from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
import os
from backend.llm_config import embeddings_llm




current_directory = os.getcwd()
db_loc = current_directory + "/db/qdrant_db"

def get_vector_store():
    """Initialize Chroma DB Vector store"""

    client = QdrantClient(path=db_loc)
    
    if not client.collection_exists("pdf_collection"):
        client.create_collection(
        collection_name="pdf_collection",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="pdf_collection",
        embedding=embeddings_llm,
    )

    return vector_store

def load_and_index_pdf(pdf_path: str):
    """Load PDF and create embeddings"""
    try:
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        all_chunks = text_splitter.split_documents(documents)

        uuids = [str(uuid4()) for _ in range(len(all_chunks))]
        

        vector_store = get_vector_store()
        vector_store.add_documents(documents=documents, ids=uuids)
        
        
        return {"status": "success", "chunks": len(all_chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant context from vector store"""
    
    try:
        results = get_vector_store().similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        return f"Retrieval error: {str(e)}"

def empty_vector_store():
    try:
        client = QdrantClient(path=db_loc)

        delete_result = client.delete(
            collection_name="pdf_collection",
            points_selector=models.FilterSelector(
                filter=models.Filter()
            ),
            wait=True  
        )
        return "Vector Store Emptied"
    except Exception as e:
        return f"Deletion error: {str(e)}"


if __name__ =='__main__':
    # print(load_and_index_pdf("temp.pdf"))
    # print(retrieve_context("whats this document about?", 3))