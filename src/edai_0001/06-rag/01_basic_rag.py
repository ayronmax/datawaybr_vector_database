from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from qdrant_client import QdrantClient

# Initialize OpenAI
embedding_model = OpenAIEmbeddings()
chat = ChatOpenAI(model='gpt-4')

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "amazon-text"

# Qdrant client initialization
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_relevant_products(query: str, limit: int = 5, score_threshold: float = 0.60):
    query_emb = embedding_model.embed_query(query)
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_emb,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True
    )
    return results  

def format_product_info(results):
    return "\n".join([
        f"name: {x.payload['Name']} Image: {x.payload['Image']}"
        for x in results
    ])

def generate_augmented_prompt(query: str, source_knowledge: str):
    return f"""Usando este contexto, responda a pergunta:
    Forneça os nomes e as imagens dos produtos. Use texto sem markdown.
    Caso não saiba a resposta, diga que não sabe.
    Contexto: {source_knowledge}
    Pergunta: {query}"""

def get_ai_response(query: str):
    relevant_products = get_relevant_products(query)
    source_knowledge = format_product_info(relevant_products)
    augmented_prompt = generate_augmented_prompt(query, source_knowledge)

    messages = [HumanMessage(content=augmented_prompt)]
    response = chat.invoke(messages)
    return response.content

if __name__ == "__main__":
    query = "Me sugira alguns funkos da marvel para comprar"
    result = get_ai_response(query)
    print(result)