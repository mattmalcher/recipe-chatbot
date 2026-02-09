"""FTS, vector, and hybrid search with the HybridRetriever class."""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from pydantic import TypeAdapter
from sqlalchemy import text
from sqlmodel import Session, select, Sequence
from tqdm import tqdm

from .db import (
    DEFAULT_DB_PATH,
    EMBEDDING_DIM,
    create_fts_index,
    create_vector_index,
    get_connection,
    get_engine,
    init_db,
)
from .embeddings import generate_embeddings, generate_query_embedding
from .models import Recipe, SearchResult, RecipeBase


# ---------------------------------------------------------------------------
# Recipe loading
# ---------------------------------------------------------------------------


def load_recipes(recipes_path: Path) -> list[RecipeBase]:
    """Load and parse recipes from JSON file using Pydantic validation.

    Args:
        recipes_path: Path to the JSON file containing recipe data.

    Returns:
        List of validated RecipeBase instances.
    """
    with open(recipes_path) as f:
        recipes_data = json.load(f)

    # Use TypeAdapter for efficient batch validation
    adapter = TypeAdapter(list[RecipeBase])
    return adapter.validate_python(recipes_data)


# ---------------------------------------------------------------------------
# Low-level search helpers (return id + score tuples for ranking)
# ---------------------------------------------------------------------------


def _fts_search(
    session: Session,
    query: str,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Full-text search via DuckDB FTS extension (BM25 scoring)."""
    rows = session.execute(  # ty:ignore[deprecated]
        text(
            "SELECT id, score FROM ("
            "  SELECT id, fts_main_recipes.match_bm25(id, :query, fields := 'full_text') AS score"
            "  FROM recipes"
            ") sq "
            "WHERE score IS NOT NULL "
            "ORDER BY score DESC "
            "LIMIT :top_k"
        ),
        {"query": query, "top_k": top_k},
    ).fetchall()

    return [(row[0], float(row[1])) for row in rows]


def _vector_search(
    session: Session,
    query_embedding: List[float],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Vector similarity search via DuckDB VSS extension (cosine distance)."""
    emb_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    rows = session.execute(  # ty:ignore[deprecated]
        text(
            f"SELECT id, "
            f"       array_cosine_distance(embedding, {emb_literal}::FLOAT[{EMBEDDING_DIM}]) AS distance "
            f"FROM recipes "
            f"WHERE embedding IS NOT NULL "
            f"ORDER BY distance ASC "
            f"LIMIT :top_k"
        ),
        {"top_k": top_k},
    ).fetchall()

    return [(row[0], 1.0 - float(row[1])) for row in rows]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *rankings: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Combine ranked lists using Reciprocal Rank Fusion.

    Each ranking is a list of (doc_id, score) tuples, ordered by relevance.
    Returns fused (doc_id, rrf_score) tuples sorted by RRF score descending.
    """
    scores: Dict[int, float] = {}

    for ranking in rankings:
        for rank_pos, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_pos)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in fused]


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
        self.engine = get_engine(db_path)
        self.rrf_k = rrf_k
        self.fts_top_k = fts_top_k
        self.vector_top_k = vector_top_k
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_and_index(self, recipes_path: Path) -> None:
        """One-time setup: load recipes into DuckDB, embed, and index."""
        # Load and validate recipes from JSON
        recipes: list[RecipeBase] = load_recipes(recipes_path)

        # Initialise DB + table
        init_db(self.db_path)
        self.engine.dispose()
        conn = get_connection(self.db_path)

        # Insert recipes
        print(f"Inserting {len(recipes)} recipes into DuckDB …")
        for recipe in tqdm(recipes, desc="Inserting recipes"):
            conn.execute(
                """
                INSERT OR REPLACE INTO recipes
                    (id, name, description, minutes, n_ingredients, n_steps,
                     submitted, contributor_id, full_text,
                     ingredients, steps, tags, nutrition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    recipe.id,
                    recipe.name,
                    recipe.description or "",
                    recipe.minutes,
                    recipe.n_ingredients,
                    recipe.n_steps,
                    recipe.submitted,
                    recipe.contributor_id,
                    recipe.full_text or "",
                    json.dumps(recipe.ingredients or []),
                    json.dumps(recipe.steps or []),
                    json.dumps(recipe.tags or []),
                    json.dumps(recipe.nutrition or {}),
                ],
            )

        # Generate embeddings
        print("Generating embeddings …")
        texts = [r.to_embedding_text() for r in recipes]
        embeddings = generate_embeddings(texts, model=self.embedding_model)

        # Store embeddings
        print("Storing embeddings …")
        for recipe, emb in tqdm(
            zip(recipes, embeddings), total=len(recipes), desc="Storing embeddings"
        ):
            emb_literal = "[" + ",".join(str(x) for x in emb) + "]"
            conn.execute(
                f"UPDATE recipes SET embedding = {emb_literal}::FLOAT[{EMBEDDING_DIM}] "
                f"WHERE id = ?",
                [recipe.id],
            )

        conn.close()

        # Build indexes
        print("Creating FTS index …")
        create_fts_index(self.db_path)
        print("Creating HNSW vector index …")
        create_vector_index(self.db_path)
        print("Done.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "fts", "vector"] = "hybrid",
    ) -> List[SearchResult]:
        """Retrieve recipes by *mode*: ``hybrid``, ``fts``, or ``vector``."""
        with Session(self.engine) as session:
            # Search → (id, score) tuples
            if mode == "fts":
                id_scores = _fts_search(session, query, top_k=top_k)
            elif mode == "vector":
                q_emb = generate_query_embedding(query, model=self.embedding_model)
                id_scores = _vector_search(session, q_emb, top_k=top_k)
            else:
                fts_results = _fts_search(session, query, top_k=self.fts_top_k)
                q_emb = generate_query_embedding(query, model=self.embedding_model)
                vec_results = _vector_search(session, q_emb, top_k=self.vector_top_k)
                id_scores = reciprocal_rank_fusion(
                    fts_results, vec_results, k=self.rrf_k
                )[:top_k]

            # Fetch full Recipe objects via ORM
            ids: list[int] = [doc_id for doc_id, _ in id_scores]
            if not ids:
                return []
            recipes: Sequence[Recipe] = session.exec(
                select(Recipe).where(Recipe.id.in_(ids))  # type: ignore[arg-type]
            ).all()
            recipe_map: dict[int, Recipe] = {r.id: r for r in recipes}

        # Build result dicts preserving search ranking order
        results: List[SearchResult] = []
        for rank, (doc_id, score) in enumerate(id_scores, start=1):
            recipe: Recipe | None = recipe_map.get(doc_id)
            if recipe is None:
                continue
            d = SearchResult(**recipe.model_dump(exclude={"embedding"}), rank=rank, bm25_score=score)
            results.append(d)
        return results

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Compatibility alias used by ``BaseRetrievalEvaluator``."""
        return [x.model_dump() for x in self.retrieve(query, top_k=top_k, mode="fts")]
