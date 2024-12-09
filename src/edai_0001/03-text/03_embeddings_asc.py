# Modificado por Alexinaldo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

# Set up OpenAI API key (ensure it's properly set in your environment)

# Load the CSV file
csv_file_path = "./data/images/home/sdf/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"
df = pd.read_csv(csv_file_path)

# Extract the 'Product Name' column
product_names = df['Product Name'].fillna('').tolist()

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(product_names, convert_to_tensor=False)

# Convert embeddings to list before adding to dataframe
df['embeddings'] = embeddings.tolist()

# Convert to Parquet
output_parquet_file = "./data/images/home/sdf/amazon_product_embeddings.parquet"
df.to_parquet(output_parquet_file, engine='pyarrow')

print(f"Embeddings saved as Parquet file: {output_parquet_file}")
