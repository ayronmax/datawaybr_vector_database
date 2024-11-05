import pandas as pd
import numpy as np

from typing import List, Optional
from fastembed import ImageEmbedding
import concurrent.futures


model = ImageEmbedding(model_name="Qdrant/Unicom-ViT-B-32")

def read_dataframe(path:str) -> pd.DataFrame:
    dataset_df = pd.read_csv(path)
    return dataset_df

def calculate_embedding(image_path: str) -> Optional[List[float]]:
    try:
        return next(model.embed([image_path])).tolist()
    except:
        return None

def parallel_apply_calculate_embedding(df: pd.DataFrame, column_name: str, num_workers: int = 8) -> pd.Series:
    print(f"Calculating Embeddings")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
      results = list(executor.map(calculate_embedding, df[column_name].astype(str)))

    return pd.Series(results)

dataset_df = read_dataframe('data/images/home/sdf/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv')
dataset_df = dataset_df.dropna(subset=["LocalImage"])

# Example usage
dataset_df["Embedding"] = parallel_apply_calculate_embedding(dataset_df, "LocalImage").replace({None: np.nan})
dataset_df = dataset_df.dropna(subset=["Embedding"])
dataset_df.to_parquet("./data/amazon-with-embeddings.parquet")