import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# Streamlit config
st.set_page_config(layout="wide")
st.title("Vector DB`s - Pesquisa por texto")

st.write('Tente: Ironman, Spiderman, Batman')
client = QdrantClient(url="http://localhost:6333") 

# Initialize SentenceTransformer model instead of OpenAI
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

search_term = st.text_input("Termos para pesquisa")

if search_term:
    # Create embeddings using SentenceTransformer
    query_embeddings = embedding_model.encode(search_term, convert_to_tensor=False)

    # Debug print
    st.write("Query embeddings shape:", query_embeddings.shape)

    search_results = client.search(
        collection_name="amazon-text",
        query_vector=query_embeddings.tolist(),  # Convert numpy array to list
        limit=3,
        score_threshold=0.3,  # Reduced threshold
        with_payload=True
    )

    # Debug print
    st.write("Número de resultados encontrados:", len(search_results))

    if not search_results:
        st.warning("Nenhum resultado encontrado. Tente reduzir o score_threshold ou usar termos diferentes.")

    for result in search_results:
        st.image(result.payload["Image"], caption=f"Score: {result.score:.4f}")
        st.write(result)
