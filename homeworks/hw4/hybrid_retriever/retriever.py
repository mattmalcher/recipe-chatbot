"""FTS, vector, and hybrid search with the HybridRetriever class."""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

import duckdb
from tqdm import tqdm

from .db import (
    DEFAULT_DB_PATH,
    EMBEDDING_DIM,
    create_fts_index,
    create_vector_index,
    get_connection,
    init_db,
)
from .embeddings import generate_embeddings, generate_query_embedding


# ---------------------------------------------------------------------------
# Low-level search helpers
# ---------------------------------------------------------------------------


def _fts_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Full-text search via DuckDB FTS extension (BM25 scoring)."""
    rows = conn.execute(
        """
        SELECT id, name, score
        FROM (
            SELECT *, fts_main_recipes.match_bm25(id, ?, fields := 'full_text') AS score
            FROM recipes
        ) sq
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        [query, top_k],
    ).fetchall()

    return [
        {"id": row[0], "name": row[1], "fts_score": float(row[2]), "fts_rank": rank + 1}
        for rank, row in enumerate(rows)
    ]


def _vector_search(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: List[float],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Vector similarity search via DuckDB VSS extension (cosine distance)."""
    emb_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    rows = conn.execute(
        f"""
        SELECT id, name,
               array_cosine_distance(embedding, {emb_literal}::FLOAT[{EMBEDDING_DIM}]) AS distance
        FROM recipes
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT {top_k}
        """
    ).fetchall()

    return [
        {
            "id": row[0],
            "name": row[1],
            "vector_distance": float(row[2]),
            "vector_score": 1.0 - float(row[2]),
            "vector_rank": rank + 1,
        }
        for rank, row in enumerate(rows)
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *rankings: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Combine ranked lists using Reciprocal Rank Fusion.

    rrf_score(doc) = Σ  1 / (k + rank_i)  over all ranking lists.
    """
    scores: Dict[int, float] = {}
    metadata: Dict[int, Dict[str, Any]] = {}

    for ranking in rankings:
        for rank_pos, item in enumerate(ranking, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_pos)
            if doc_id not in metadata:
                metadata[doc_id] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**metadata[doc_id], "rrf_score": score, "hybrid_rank": rank + 1}
        for rank, (doc_id, score) in enumerate(fused)
    ]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combined FTS + vector retriever backed by DuckDB.

    Compatible with ``BaseRetrievalEvaluator`` via the ``retrieve_bm25`` method.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        rrf_k: int = 60,
        fts_top_k: int = 20,
        vector_top_k: int = 20,
        embedding_model: str = "text-embedding-3-large",
    ):
        self.db_path = db_path
        self.rrf_k = rrf_k
        self.fts_top_k = fts_top_k
        self.vector_top_k = vector_top_k
        self.embedding_model = embedding_model

        # Loaded recipe dicts (for enriching results)
        self.recipes: List[Dict[str, Any]] = []
        self._recipe_lookup: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_and_index(self, recipes_path: Path) -> None:
        """One-time setup: load recipes into DuckDB, embed, and index."""
        with open(recipes_path) as f:
            self.recipes = json.load(f)
        self._recipe_lookup = {r["id"]: r for r in self.recipes}

        # Initialise DB + table
        init_db(self.db_path)
        conn = get_connection(self.db_path)

        # Insert recipes
        print(f"Inserting {len(self.recipes)} recipes into DuckDB …")
        for recipe in tqdm(self.recipes, desc="Inserting recipes"):
            conn.execute(
                """
                INSERT OR REPLACE INTO recipes
                    (id, name, description, minutes, n_ingredients, n_steps,
                     submitted, contributor_id, full_text,
                     ingredients_text, steps_text, tags_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    recipe["id"],
                    recipe["name"],
                    recipe.get("description", ""),
                    recipe.get("minutes", 0),
                    recipe.get("n_ingredients", 0),
                    recipe.get("n_steps", 0),
                    recipe.get("submitted"),
                    recipe.get("contributor_id"),
                    recipe.get("full_text", ""),
                    " ".join(recipe.get("ingredients", [])),
                    " ".join(recipe.get("steps", [])),
                    " ".join(recipe.get("tags", [])),
                ],
            )

        # Generate embeddings
        print("Generating embeddings …")
        texts = [r.get("full_text", "") for r in self.recipes]
        embeddings = generate_embeddings(texts, model=self.embedding_model)

        # Store embeddings
        print("Storing embeddings …")
        for recipe, emb in tqdm(
            zip(self.recipes, embeddings), total=len(self.recipes), desc="Storing embeddings"
        ):
            emb_literal = "[" + ",".join(str(x) for x in emb) + "]"
            conn.execute(
                f"UPDATE recipes SET embedding = {emb_literal}::FLOAT[{EMBEDDING_DIM}] "
                f"WHERE id = ?",
                [recipe["id"]],
            )

        conn.close()

        # Build indexes
        print("Creating FTS index …")
        create_fts_index(self.db_path)
        print("Creating HNSW vector index …")
        create_vector_index(self.db_path)
        print("Done.")

    def load_recipes(self, recipes_path: Path) -> None:
        """Load recipe metadata for result enrichment (no DB write)."""
        with open(recipes_path) as f:
            self.recipes = json.load(f)
        self._recipe_lookup = {r["id"]: r for r in self.recipes}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "fts", "vector"] = "hybrid",
    ) -> List[Dict[str, Any]]:
        """Retrieve recipes by *mode*: ``hybrid``, ``fts``, or ``vector``."""
        conn = get_connection(self.db_path)

        try:
            if mode == "fts":
                raw = _fts_search(conn, query, top_k=top_k)
            elif mode == "vector":
                q_emb = generate_query_embedding(query, model=self.embedding_model)
                raw = _vector_search(conn, q_emb, top_k=top_k)
            else:
                fts_results = _fts_search(conn, query, top_k=self.fts_top_k)
                q_emb = generate_query_embedding(query, model=self.embedding_model)
                vec_results = _vector_search(conn, q_emb, top_k=self.vector_top_k)
                raw = reciprocal_rank_fusion(fts_results, vec_results, k=self.rrf_k)[
                    :top_k
                ]
        finally:
            conn.close()

        return self._enrich(raw, top_k)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Compatibility alias used by ``BaseRetrievalEvaluator``."""
        return self.retrieve(query, top_k=top_k, mode="hybrid")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enrich(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Enrich raw search results with full recipe data."""
        enriched: List[Dict[str, Any]] = []
        for i, result in enumerate(results[:top_k]):
            recipe_data = self._recipe_lookup.get(result["id"], {})
            merged = {**recipe_data, **result}
            merged["rank"] = i + 1
            # BaseRetrievalEvaluator reads 'bm25_score'
            merged["bm25_score"] = result.get(
                "rrf_score",
                result.get("fts_score", result.get("vector_score", 0.0)),
            )
            enriched.append(merged)
        return enriched
