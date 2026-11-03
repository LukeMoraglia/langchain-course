import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("INDEX_NAME")

# 1. Delete the old document-schema index if it exists
if pc.has_index(index_name):
    print(f"Deleting existing index: {index_name}")
    pc.delete_index(index_name)

# 2. Create a standard vector index (spec-based, non-document schema)
print(f"Creating standard vector index: {index_name}")
pc.create_index(
    name=index_name,
    dimension=1536,  # Must match text-embedding-3-small / text-embedding-ada-002
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # Match your preferred region (e.g., us-east-1, us-west-2)
    )
)
print("Standard index created successfully!")