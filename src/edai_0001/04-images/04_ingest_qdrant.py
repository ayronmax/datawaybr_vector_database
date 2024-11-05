import pandas as pd
import uuid


from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

client = QdrantClient("localhost")
collections = client.get_collections()
print(collections)

client.create_collection(
    collection_name="amazon-images",
    vectors_config=rest.VectorParams(
        size=512,
        distance=rest.Distance.COSINE,
    )
)

file_path = './data/amazon-with-embeddings.parquet'
dataset_df = pd.read_parquet(file_path)

payloads = (
  dataset_df[["Uniq Id", "Product Name", "About Product", "Image", "LocalImage"]]
    .fillna("Unknown")
    .rename(columns={"Uniq Id": "ID",
                     "Product Name": "Name",
                     "About Product": "Description",
                     "LocalImage": "Path"})
    .to_dict("records")
)

client.upload_collection(
    collection_name="amazon-images",
    vectors=list(map(list, dataset_df["Embedding"].tolist())),
    payload=payloads,
    ids=[uuid.uuid4().hex for _ in payloads],
)

# Quantidade de Registros
print(client.count("amazon-images"))
