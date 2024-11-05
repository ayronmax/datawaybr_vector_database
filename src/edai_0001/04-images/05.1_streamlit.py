import streamlit as st
import llm
import time
from PIL import Image
from qdrant_client import QdrantClient, models

# Streamlit config
st.set_page_config(layout="wide")
st.title("Image Search Engine")
client = QdrantClient(url="http://localhost:6333") 

# Enter search term or provide image
option = st.selectbox('How do you want to search?', ('Search Term', 'Image'))
if option == "Search Term":
    uploaded_file = None
    search_term = st.text_input("Enter search term")
else:
    search_term = None
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=200)

model = llm.get_embedding_model("clip")


if option and (uploaded_file or search_term):
    with st.spinner('Searching'):
        if option == 'Search Term':
            query_embeddings = model.embed(search_term)
        else:
            query_embeddings = model.embed(uploaded_file.getvalue())

        st.write(query_embeddings)
        search_results = client.search(
            collection_name="amazon-images",
            query_vector=query_embeddings,
            limit=5,
            #score_threshold=0.80,
            with_payload=True
        )

    for result in search_results:
        st.image(result.payload["Image"], caption=f"Score: {result.score:.4f}")
        st.write(result)