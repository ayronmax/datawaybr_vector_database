import streamlit as st

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

# Streamlit config
st.set_page_config(layout="wide")
st.title("Vector DB`s - Pesquisa por texto")

st.write('Try: Funko, Funkko, Airplane')
client = QdrantClient(url="http://localhost:6333") 

# Initialize Langchain's OpenAI Embedding model
embedding_model = OpenAIEmbeddings()

search_term = st.text_input("Termos para pesquisa")

if search_term:
    # Create embeddings for the product names
    query_embeddings = embedding_model.embed_query(search_term)

    st.write(query_embeddings)

    search_results = client.search(
        collection_name="amazon-text",
        query_vector=query_embeddings,
        limit=3,
        score_threshold=0.80,
        with_payload=True
    )

    for result in search_results:
        st.image(result.payload["Image"], caption=f"Score: {result.score:.4f}")
        st.write(result)