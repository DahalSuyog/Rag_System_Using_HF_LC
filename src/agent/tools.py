# tools.py
import logging
from langchain_core.tools import tool, BaseTool

logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINE YOUR TOOLS HERE
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


@tool
def retrieve_documents(query: str, top_k: int = 3) -> str:
    """
    Call this tool when the user asks a question that requires fetching
    information, facts, or knowledge from the document store / knowledge base.
    """
    logger.info("retrieve_documents | query=%r top_k=%d", query, top_k)
    return f"[Retrieval stub] query='{query}', top_k={top_k}."


# ==========================================
# 2. AUTOMATIC REGISTRY BACKEND
# ==========================================

def get_all_tools() -> list[BaseTool]:
    """
    Automatically grabs every LangChain tool defined in this module global scope.
    Developers only need to add @tool to their function; it will appear here instantly.
    """
    # globals().values() lets us look at everything defined in this file
    return [obj for obj in globals().values() if isinstance(obj, BaseTool) and obj.func.__module__ == __name__]