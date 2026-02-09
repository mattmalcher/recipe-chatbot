"""Shared fixtures for hybrid_retriever tests."""

import json
import random

import pytest

from sqlmodel import Session

from homeworks.hw4.hybrid_retriever.db import (
    EMBEDDING_DIM,
    create_fts_index,
    create_vector_index,
    get_connection,
    get_engine,
    init_db,
)
from homeworks.hw4.hybrid_retriever.retriever import HybridRetriever

SAMPLE_RECIPES = [
    {
        "id": 1,
        "name": "chocolate cake",
        "description": "A rich chocolate layer cake",
        "minutes": 60,
        "ingredients": ["flour", "sugar", "cocoa", "eggs", "butter"],
        "n_ingredients": 5,
        "steps": ["preheat oven", "mix dry ingredients", "add wet ingredients", "bake"],
        "n_steps": 4,
        "tags": ["dessert", "chocolate", "cake"],
        "full_text": "chocolate cake a rich chocolate layer cake flour sugar cocoa eggs butter preheat oven mix dry ingredients add wet ingredients bake dessert chocolate cake",
        "nutrition": {},
        "submitted": "2020-01-01",
        "contributor_id": 100,
    },
    {
        "id": 2,
        "name": "chicken pasta",
        "description": "Creamy garlic chicken pasta",
        "minutes": 30,
        "ingredients": ["chicken", "pasta", "garlic", "cream", "parmesan"],
        "n_ingredients": 5,
        "steps": ["cook pasta", "saute chicken", "make sauce", "combine"],
        "n_steps": 4,
        "tags": ["dinner", "pasta", "chicken"],
        "full_text": "chicken pasta creamy garlic chicken pasta chicken pasta garlic cream parmesan cook pasta saute chicken make sauce combine dinner pasta chicken",
        "nutrition": {},
        "submitted": "2020-02-01",
        "contributor_id": 101,
    },
    {
        "id": 3,
        "name": "vegetable stir fry",
        "description": "Quick and healthy vegetable stir fry with soy sauce",
        "minutes": 15,
        "ingredients": ["broccoli", "carrot", "soy sauce", "ginger", "rice"],
        "n_ingredients": 5,
        "steps": ["chop vegetables", "heat wok", "stir fry", "serve over rice"],
        "n_steps": 4,
        "tags": ["dinner", "vegetable", "healthy", "quick"],
        "full_text": "vegetable stir fry quick and healthy vegetable stir fry with soy sauce broccoli carrot soy sauce ginger rice chop vegetables heat wok stir fry serve over rice dinner vegetable healthy quick",
        "nutrition": {},
        "submitted": "2020-03-01",
        "contributor_id": 102,
    },
]


@pytest.fixture
def sample_recipes():
    return SAMPLE_RECIPES


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.duckdb"


@pytest.fixture
def seeded_db(db_path, sample_recipes):
    """DB with recipes inserted and FTS index created (no embeddings)."""
    init_db(db_path)
    conn = get_connection(db_path)
    for r in sample_recipes:
        conn.execute(
            """
            INSERT INTO recipes
                (id, name, description, minutes, n_ingredients, n_steps,
                 submitted, contributor_id, full_text,
                 ingredients, steps, tags, nutrition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                r["id"],
                r["name"],
                r.get("description", ""),
                r.get("minutes", 0),
                r.get("n_ingredients", 0),
                r.get("n_steps", 0),
                r.get("submitted"),
                r.get("contributor_id"),
                r.get("full_text", ""),
                json.dumps(r.get("ingredients", [])),
                json.dumps(r.get("steps", [])),
                json.dumps(r.get("tags", [])),
                json.dumps(r.get("nutrition", {})),
            ],
        )
    conn.close()
    create_fts_index(db_path)
    return db_path


@pytest.fixture
def seeded_session(seeded_db):
    """SQLModel Session for a seeded DB (FTS ready, no embeddings)."""
    engine = get_engine(seeded_db)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def populated_session(populated_db):
    """SQLModel Session for a fully populated DB (FTS + embeddings + HNSW)."""
    engine = get_engine(populated_db)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _make_fake_embedding(seed: int) -> list[float]:
    """Deterministic random 3072-dim vector from a seed."""
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]


@pytest.fixture
def fake_embeddings(sample_recipes):
    """Dict mapping recipe id -> deterministic fake embedding vector."""
    return {r["id"]: _make_fake_embedding(r["id"]) for r in sample_recipes}


@pytest.fixture
def populated_db(seeded_db, fake_embeddings):
    """seeded_db + fake embeddings stored + HNSW vector index created."""
    conn = get_connection(seeded_db)
    for recipe_id, emb in fake_embeddings.items():
        emb_literal = "[" + ",".join(str(x) for x in emb) + "]"
        conn.execute(
            f"UPDATE recipes SET embedding = {emb_literal}::FLOAT[{EMBEDDING_DIM}] "
            f"WHERE id = ?",
            [recipe_id],
        )
    conn.close()
    create_vector_index(seeded_db)
    return seeded_db


@pytest.fixture
def retriever(populated_db):
    """HybridRetriever backed by populated_db."""
    return HybridRetriever(db_path=populated_db)
