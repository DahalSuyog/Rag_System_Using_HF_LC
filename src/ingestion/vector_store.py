import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv




class VectorStoreManager:
    '''
    Manages the embedding and vector storeing langchain documents in Pinecone Database.
    '''
    def __init__(self, index_name: str, embedding_model: str = "text-embedding-3-small", dimension: int = 1536):
        '''
        Initializes Pinecone clients and sets up embedding models.
        
        :param index_name: Name of the Pinecone index to target or create.
        :param dimension: 1536 for OpenAI's 'text-embedding-3-small' model.
        
        '''
        self.index_name = index_name
        self.dimension = dimension
        self.embedding_model = embedding_model
        #This initializes the embedding model
        #It automatically uses the OPENAI_API_KEY from the environment variables
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

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
    # Ensure keys are loaded in your local environment for testing
    load_dotenv()  # This will load the .env file and set environment variables
    
    # Target index
    MY_INDEX = "pdf-chatbot-prototype"
    
    # 1. Initialize manager
    db_manager = VectorStoreManager(index_name=os.getenv("VECTOR_STORE_INDEX_NAME"))
    
    # 2. Mock documents mimicking the output from your PdfParser split_text_into_chunks()
    mock_chunks = [
        Document(page_content="The revenue of the company in Q3 was $5 million.", metadata={"page_number": 1}),
        Document(page_content="Our core policy states that employees can work remotely.", metadata={"page_number": 2}),
        Document(page_content="The primary contact for emergency operations is John Doe.", metadata={"page_number": 5})
    ]
    
    # 3. Store documents in the cloud
    db_manager.store_documents(mock_chunks)
    
    # 4. Test Query Retrieval
    test_query = "What was the company's revenue?"
    print(f"\n--- Testing Similarity Search for: '{test_query}' ---")
    results = db_manager.similarity_search(test_query, k=1)
    
    for doc in results:
        print(f"Found Content: {doc.page_content}")
        print(f"Source Metadata: {doc.metadata}")  # This will print: {'page_number': 1}