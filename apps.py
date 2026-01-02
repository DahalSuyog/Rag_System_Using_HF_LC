from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Simple RAG Ingestion")

#DATA MODELS 
class IngestRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# --- VECTOR STORE SETUP ---
# 1. Load the model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []  # Our "database" to store the actual text

# --- LOGIC FUNCTIONS ---

def ingest_text(text: str):
    # Encode text into a vector
    embedding = model.encode(text)
    # FAISS requires a 2D array of float32
    embedding_array = np.array([embedding]).astype("float32")
    
    index.add(embedding_array)
    documents.append(text)
    print(f"Total documents now: {len(documents)}")

def search(query: str, k=3):
    if not documents:
        return ["No documents ingested yet."]

    # Encode the query
    query_vector = model.encode(query)
    query_vector = np.array([query_vector]).astype("float32")

    # Search the index
    # distances: how far the match is (lower is better for L2)
    # indices: the position of the match in our 'documents' list
    distances, indices = index.search(query_vector, k)

    # Convert indices to actual text snippets
    results = []
    for i in indices[0]:
        if i != -1 and i < len(documents):  # -1 means no match found
            results.append(documents[i])
    
    return results

# --- API ENDPOINTS ---

@app.get("/")
def root():
    return {"message": "RAG API is running", "doc_count": len(documents)}

@app.post("/ingest")
def ingest(req: IngestRequest):
    ingest_text(req.text)
    return {"message": "Text ingested successfully", "current_count": len(documents)}

@app.post("/query")
def query(req: QueryRequest):
    results = search(req.question)
    return {"context": results}