from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from typing import List, Dict
import os

# FastAPI initialization
app = FastAPI()

# Environment variables for configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "amazon-text")

# Qdrant client initialization
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Embedding initialization
embedding_model = OpenAIEmbeddings()

# Models
class Query(BaseModel):
    query: str = Field(..., description="The search query")

class SearchResult(BaseModel):
    results: List[Dict] = Field(..., description="List of search results")

# OpenAI embedding generation function
def generate_embedding(text: str) -> List[float]:
    return embedding_model.embed_query(text)

# Search function
def search_qdrant(embedding: List[float]) -> List[Dict]:
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=5,
        score_threshold=0.80,
        with_payload=True
    )
    return [result.payload for result in search_result]

# API route for search
@app.post("/search", response_model=SearchResult)
async def search_terms(query: Query) -> SearchResult:
    try:
        embedding = generate_embedding(query.query)
        results = search_qdrant(embedding)
        return SearchResult(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# uvicorn 01_api:app --reload