import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

FILE_PATH = r"Notebook\suyog\Master_in_HM.xlsx"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "university_programs"

def search_university_data(query_text):
    # Load the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Connect to the local Qdrant database
    client = QdrantClient(path=QDRANT_PATH)
    
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings
    )

    # 3. Perform the search
    # k=3 means it will return the top 3 most relevant rows from your Excel
    print(f"\nSearching for: '{query_text}'...")
    results = vectorstore.similarity_search(query_text, k=5)
    

    # 4. Display the results nicely
    print("-" * 50)
    if not results:
        print("No matches found.")
    else:
        for i, doc in enumerate(results):
            print(f"Match #{i+1}:")
            print(f"📍 University/Program: {doc.page_content}")
            print(f"💰 Fees: {doc.metadata.get('Tuition Fees', 'N/A')}")
            print(f"📅 Deadline: {doc.metadata.get('App Deadline (next intake)', 'N/A')}")
            print(f"📝 Notes: {doc.metadata.get('Notes / DAAD link', 'N/A')}")
            print("-" * 30)
    return results