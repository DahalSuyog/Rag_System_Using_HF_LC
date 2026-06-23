import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.parser import PdfParser


class VectorStoreManager:
    '''
    Manages the embedding and vector storeing langchain documents in Pinecone Database.
    '''
    def __init__(self, index_name: str | None = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384):
        '''
        Initializes Pinecone clients and sets up embedding models.
        
        :param index_name: Name of the Pinecone index to target or create.
        :param dimension: 384 for OpenAI's 'sentence-transformers/all-MiniLM-L6-v2' model.
        
        '''
        self.index_name = index_name or os.getenv("VECTOR_STORE_INDEX_NAME", "pdf-chatbot-prototype")
        self.dimension = dimension
        self.embedding_model = embedding_model
        #This initializes the embedding model
        #It automatically uses the OPENAI_API_KEY from the environment variables
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # 2. Initialize the core Pinecone client
        # (This automatically looks for your PINECONE_API_KEY environment variable)
        self.pc = Pinecone()

        #This ensures index is present in pinecone serverless environment, if not it creates one
        self._ensure_index_exists()

        # 4. Bind the index connection to LangChain's Pinecone interface
        self.pinecone_index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(
            index=self.pinecone_index, 
            embedding=self.embeddings,
            text_key="text"  # Field name where raw text is saved inside Pinecone metadata
        )

    def _ensure_index_exists(self):
        """
        Internal helper method to check if the index exists. 
        If not, it automatically creates a cloud serverless instance.

        """
        if not self.pc.has_index(self.index_name):
            print(f"Index '{self.index_name}' not found. Creating a serverless index...")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Free-tier serverless standard region
                )
            )
            print(f"Successfully created index '{self.index_name}'.")
        else:
            print(f"Index '{self.index_name}' already exists. Connecting...")

    def store_documents(self, documents: List[Document]) -> List[str]:
        """
        Embeds documents and upserts them into Pinecone. 
        Preserves all metadata like page_number.
        
        :param documents: List of LangChain Document objects from PdfParser.
        :return: List of generated ID strings from Pinecone.
        """
        if not documents:
            print("No documents provided to store.")
            return []

        print(f"Embedding and storing {len(documents)} document chunks into Pinecone...")
        ids = self.vector_store.add_documents(documents=documents)
        print("Upsert completed successfully.")
        return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Queries Pinecone to find the most contextually relevant document chunks.
        
        :param query: The user's text question.
        :param k: Number of chunks to return.
        :return: List of matched LangChain Documents containing text and metadata.
        """
        return self.vector_store.similarity_search(query=query, k=k)
    
    # ==========================================
# Example Usage / Local Integration Test
# ==========================================
if __name__ == "__main__":
    # Load environment variables (API keys, Pinecone keys, etc.)
    load_dotenv()  
    
    # Target index (Fallback to "pdf-chatbot-prototype" if env var is missing)
    MY_INDEX = os.getenv("VECTOR_STORE_INDEX_NAME", "pdf-chatbot-prototype")
    
    # 1. Initialize manager and parser
    db_manager = VectorStoreManager(index_name=MY_INDEX)
    parser = PdfParser(chunk_size=500, chunk_overlap=100)
    
    # Path to your real PDF
    sample_pdf = "/home/suyogdahal/Desktop/project/Rag_System_Using_HF_LC/temp_uploads/saap_company_budget_rag_test.pdf"
    
    try:
        # 2. Extract and chunk real PDF data
        print(f"--- Phase 1: Extracting text from {sample_pdf} ---")
        raw_pages = parser.extract_text_by_page(sample_pdf)
        print(f"Successfully processed {len(raw_pages)} pages.\n")
        
        print("--- Phase 2: Splitting text into chunks ---")
        processed_chunks = parser.split_text_into_chunks(raw_pages)
        print(f"Generated {len(processed_chunks)} total chunks.\n")
        
        # 3. Store the REAL documents in the cloud vector store
        print("--- Phase 3: Storing documents in Vector DB ---")
        db_manager.store_documents(processed_chunks)
        print(f"Successfully stored {len(processed_chunks)} chunks in {MY_INDEX}.\n")
        
        # 4. Test Query Retrieval
        # Tip: Adjust this query based on what you know is actually inside your PDF!
        test_query = "What was the company's revenue?"
        print(f"--- Phase 4: Testing Similarity Search for: '{test_query}' ---")
        
        # Fetch the top 1 most similar chunk
        results = db_manager.similarity_search(test_query, k=1)
        
        # 5. Inspect the results
        if results:
            for doc in results:
                print("\n✅ --- Search Result Found ---")
                print(f"Found Content: {doc.page_content}")
                print(f"Source Metadata: {doc.metadata}") 
        else:
            print("❌ No results returned from the vector store.")
            
    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")