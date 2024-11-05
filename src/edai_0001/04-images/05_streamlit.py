import streamlit as st
from fastembed import ImageEmbedding
from PIL import Image
from qdrant_client import QdrantClient, models
import os

# Initialize Qdrant Client
client = QdrantClient(url="http://localhost:6333")  # Adjust the URL if using a different host

# Load Image Embedding Model
model = ImageEmbedding(model_name="Qdrant/Unicom-ViT-B-32")

# Streamlit UI for uploading an image
st.sidebar.title("Busca por similaridade (Imagens)")
uploaded_file = st.sidebar.file_uploader("Carregue uma imagem:", type=["png", "jpg", "jpeg"])

# Function to convert image to embeddings
def get_image_embedding(image):
    image_embedding = next(model.embed(image)).tolist()
    return image_embedding

# If an image is uploaded
if uploaded_file is not None:
    # Convert the image to embeddings
    image = Image.open(uploaded_file)
    embeddings = get_image_embedding(image)
    st.write(image)

    # Query Qdrant for similar images
    def query_qdrant(embedding, top_k=5):
        search_results = client.search(
          collection_name="amazon-images",
          query_vector=embedding,
          limit=top_k,
          score_threshold=0.80,
          with_payload=True
      )
        return search_results

    st.write("Calculating embeddings and searching for similar images...")
    
    # Get top 5 similar images from Qdrant
    search_results = query_qdrant(embeddings)
    
    # Display the results
    for result in search_results:
        st.image(result.payload["Image"], caption=f"Score: {result.score:.4f}")
        st.write(result)