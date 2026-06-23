import streamlit as st
import os
from pathlib import Path

# Import your custom modules based on your folder structure
from src.ingestion import PdfParser
from src.ingestion import VectorStoreManager
from src.agent import Model


# Optional: If your Model class doesn't automatically load tools internally, 
# you can import them here.
# from src.inference.tools import get_all_tools 

# Temporary directory to save uploaded PDFs
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize your core classes
@st.cache_resource
def init_rag_components():
    parser = PdfParser()
    vector_store = VectorStoreManager()
    model = Model(vector_store=vector_store)
    return parser, vector_store, model

parser, vector_store, model = init_rag_components()

# --- STREAMLIT UI ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise RAG System")
st.caption("Upload a PDF to populate the knowledge base, then ask questions.")

# Sidebar for file uploads
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = UPLOAD_DIR / uploaded_file.name
        
        # Save file to disk so your parser can read it via file_path string
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(file_path.name)
        
        # Trigger Ingestion Flow
        if st.button("Process & Index Document", type="primary"):
            with st.spinner("Parsing PDF and generating embeddings..."):
                try:
                    # 1. Parse PDF pages
                    pages_data = parser.extract_text_by_page(str(file_path))
                    
                    # 2. Chunk text
                    chunks = parser.split_text_into_chunks(pages_data)
                    
                    # 3. Convert dict chunks to Document objects expected by your vector store
                    # (Adjust the Document import/instantiation according to your exact framework)
                    from langchain_core.documents import Document 
                    
                    documents = [
                        Document(
                            page_content=chunk.page_content, 
                            metadata={"page": chunk.metadata.get("page_number", 0), "source": uploaded_file.name}
                        ) 
                        for chunk in chunks
                    ]
                    
                    # 4. Store chunks into Vector DB
                    vector_store.store_documents(documents)
                    
                    st.success("✨ Document successfully indexed! Ready to chat.")
                    
                    # Clean up file after ingestion if desired
                    os.remove(file_path)
                except Exception as e:
                    st.error(f"An error occurred during ingestion: {e}")

# --- CHAT INTERFACE ---
# Display historical chat messages
for msg in st.session_state.messages:
    # Basic check to see if the message object has a content property (LangChain style)
    # or fallback to string checking if you stored raw dictionaries
    role = "user" if getattr(msg, "type", None) == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(getattr(msg, "content", str(msg)))

# Accept user input
if user_query := st.chat_input("Ask something about your document..."):
    
    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Append to state (Assuming you're passing/storing LangChain BaseMessage types based on your hints)
    from langchain_core.messages import HumanMessage, AIMessage
    st.session_state.messages.append(HumanMessage(content=user_query))
    
    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Run the model with user query and context history
                # Note: Your model will likely perform similarity_search internally using vector_store
                response_text = model.run(
                    user_message=user_query, 
                    history=st.session_state.messages[:-1] # Exclude the current message if handled by run()
                )
                
                st.markdown(response_text)
                st.session_state.messages.append(AIMessage(content=response_text))
                
            except Exception as e:
                st.error(f"Error generating response: {e}")