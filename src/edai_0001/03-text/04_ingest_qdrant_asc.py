import pandas as pd
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient("localhost")
collections = client.get_collections()
print(collections)

client.create_collection(
    collection_name="amazon-text",
    vectors_config=rest.VectorParams(
        size=384,
        distance=rest.Distance.COSINE,
    )
)

file_path = './data/images/home/sdf/amazon_product_embeddings.parquet'
dataset_df = pd.read_parquet(file_path)

payloads = (
  dataset_df[["Uniq Id", "Product Name", "Product Description", "Image"]]
    .fillna("Unknown")
    .rename(columns={"Uniq Id": "ID",
                     "Product Name": "Name",
                     "Product Description": "Description"})
    .to_dict("records")
)

client.upload_collection(
    collection_name="amazon-text",
    vectors=list(map(list, dataset_df["embeddings"].tolist())),
    payload=payloads,
    ids=[uuid.uuid4().hex for _ in payloads],
)

# Quantidade de Registros
print(client.count("amazon-text"))
