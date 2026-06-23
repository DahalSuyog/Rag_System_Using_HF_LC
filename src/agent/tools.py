# tools.py
import logging
from typing import List
from langchain_core.tools import tool, BaseTool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ==========================================
# 1. STATIC TOOLS (No dependencies needed)
# ==========================================

@tool
def greet_user(name: str = "") -> str:
    """
    Call this tool when the user is greeting you (e.g. hello, hi, hey,
    good morning, how are you). Pass the user's name if they mentioned it,
    otherwise leave it empty.
    """
    if name:
        return f"Hello, {name}! How can I help you today?"
    return "Hello! How can I help you today?"


# ==========================================
# 2. DYNAMIC TOOL FACTORY (Injects Vector Store)
# ==========================================

def get_all_tools(vector_store=None) -> list[BaseTool]:
    """
    Returns a list of tools. 
    Accepts a vector_store instance to inject into the retrieval tool.
    """
    
    # We define the tool INSIDE this function so it has access to `vector_store`
    @tool
    def retrieve_documents(query: str) -> str:
        """
        Use this tool to search the knowledge base and retrieve relevant documents.
        Pass the user's question as the 'query' argument.
        """
        if vector_store is None:
            return "Error: Vector store is not connected."
        
        try:
            # Perform similarity search
            docs = vector_store.similarity_search(query, k=3)
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return f"Error retrieving documents: {e}"
        
        # Format the output so the LLM can read it easily
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            page_num = doc.metadata.get("page_number", "N/A")
            formatted_docs.append(f"Document {i+1} (Page {page_num}):\n{doc.page_content}")
            
        return "\n\n".join(formatted_docs)

    # Return the static tools + the dynamically generated retrieval tool
    return [greet_user, retrieve_documents]