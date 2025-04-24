"""
Vector store RAG layer (Weaviate Cloud Free tier)
"""
import os, weaviate, hashlib

_WEAVIATE_URL = os.environ["WEAVIATE_URL"]
_WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

client = weaviate.Client(
    url=_WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(_WEAVIATE_API_KEY),
    additional_headers={"X-OpenAI-Api-Key": "placeholder"},
)

def semantic_retrieval(query: str, k: int = 4) -> str:
    resp = (
        client.query
        .get("PubMedPaper", ["title", "abstract"])
        .with_near_text({"concepts": [query]})
        .with_limit(k)
        .do()
    )
    items = resp["data"]["Get"]["PubMedPaper"]
    return "\n\n".join(f"{it['title']}\n{it['abstract']}" for it in items) 