"""Embedding generation via litellm."""

import time
from typing import List

import litellm
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 50


def generate_embeddings(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE,
) -> List[List[float]]:
    """Generate embeddings for a list of texts in batches."""
    all_embeddings: List[List[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]

        response = litellm.embedding(model=model, input=batch, dimensions=EMBEDDING_DIM)
        batch_embeddings = [item["embedding"] for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Rate-limit safety between batches
        if i + batch_size < len(texts):
            time.sleep(0.1)

    return all_embeddings


def generate_query_embedding(
    query: str,
    model: str = EMBEDDING_MODEL,
) -> List[float]:
    """Generate an embedding for a single query string."""
    response = litellm.embedding(model=model, input=[query], dimensions=EMBEDDING_DIM)
    return response.data[0]["embedding"]
