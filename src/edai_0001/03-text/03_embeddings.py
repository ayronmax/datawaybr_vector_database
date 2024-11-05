import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from langchain_openai import OpenAIEmbeddings

# Set up OpenAI API key (ensure it's properly set in your environment)

# Load the CSV file
csv_file_path = "./data/images/home/sdf/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"
df = pd.read_csv(csv_file_path)

# Extract the 'Product Name' column
product_names = df['Product Name'].fillna('').tolist()

# Initialize Langchain's OpenAI Embedding model
embedding_model = OpenAIEmbeddings()

# Create embeddings for the product names
embeddings = embedding_model.embed_documents(product_names)

# Add embeddings to the dataframe
df['embeddings'] = embeddings

# Convert to Parquet
output_parquet_file = "./data/images/home/sdf/amazon_product_embeddings.parquet"
df.to_parquet(output_parquet_file, engine='pyarrow')

print(f"Embeddings saved as Parquet file: {output_parquet_file}")