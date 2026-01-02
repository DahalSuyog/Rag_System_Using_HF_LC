import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qwen_inference_try import agentho as ag
from utils import search_university_data


FILE_PATH = r"Notebook\suyog\Master_in_HM.xlsx"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "university_programs"

def run_ingestion():
    #Loading and Cleaning Excel using Pandas
    print("Loading Excel file...")
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    df = pd.read_excel(FILE_PATH)
    df = df.fillna("N/A")
    df.columns = df.columns.str.strip()

    # Convert to Documents for LangChain
    # Content: University name | Metadata: Fees, Deadlines, etc.
    loader = DataFrameLoader(df, page_content_column="University / Programme (link)")
    documents = loader.load()

    # Chunking the Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    # Hugging Face Embeddings
    print("Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Setup Qdrant Store
    print("Connecting to Qdrant and indexing...")
    
    # We use a local path for persistence
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True  # Set to False if you want to keep adding to the same DB
    )

    print(f"Success! Local Qdrant index created at: {QDRANT_PATH}")

#run_ingestion()

# Function to check Qdrant status

def check_qdrant_status():
    print(f"--- Checking Qdrant Status ---")
    
    # Check if the physical folder exists
    if os.path.exists(QDRANT_PATH):
        print(f" Folder found: '{QDRANT_PATH}' exists.")
    else:
        print(f" Folder missing: '{QDRANT_PATH}' does not exist.")
        return

    try:
        # Connect to the local client
        client = QdrantClient(path=QDRANT_PATH)
        collections_response = client.get_collections()
        collection_names = [c.name for c in collections_response.collections]
        
        if COLLECTION_NAME in collection_names:
            # Get collection info
            info = client.get_collection(collection_name=COLLECTION_NAME)
            
            # FIXED: Use points_count instead of vectors_count
            print(f"✅ Collection found: '{COLLECTION_NAME}' is ready.")
            print(f"📊 Statistics: Found {info.points_count} records in the database.")
            
            if info.points_count > 0:
                print("You are ready to search!")
            else:
                print("The database is empty. You need to run ingestion.")
        else:
            print(f"Collection '{COLLECTION_NAME}' not found in the DB.")
            
    except Exception as e:
        print(f"Connection error: {e}")

check_qdrant_status()



def user_query_input():
    query = input("Enter your query about university programs: ")
    return query

if __name__ == "__main__":
    # Check if the database folder exists
    if not os.path.exists(QDRANT_PATH):
        print("--- First time setup: Ingesting data ---")
        run_ingestion()
    
    # Now ask for the query
    user_query = user_query_input()

    # Add a check to ensure we don't try to search if ingestion failed
    try:
        
        search_university_data(user_query)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Try running the 'run_ingestion()' function manually once.")

    try:
        ag(user_query)
    except Exception as e:
        print(f"An error occurred while running the agent: {e}")

        