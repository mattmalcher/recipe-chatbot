"""Database engine, initialization, and index creation for DuckDB."""

from pathlib import Path

import duckdb
from sqlalchemy import event
from sqlmodel import SQLModel, create_engine

from .models import EMBEDDING_DIM, Recipe  # noqa: F401 – ensure table is registered

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes.duckdb"


def get_engine(db_path: Path = DEFAULT_DB_PATH):
    """Create a SQLAlchemy engine backed by DuckDB.

    FTS, VSS, and HNSW persistence are configured via a connection event
    so that every connection from the pool has extensions loaded.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"duckdb:///{db_path}")

    @event.listens_for(engine, "connect")
    def _load_extensions(dbapi_conn, connection_record):
        dbapi_conn.execute("INSTALL fts; LOAD fts;")
        dbapi_conn.execute("INSTALL vss; LOAD vss;")
        dbapi_conn.execute("SET hnsw_enable_experimental_persistence = true;")

    return engine


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Get a raw DuckDB connection for native operations."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("INSTALL fts; LOAD fts;")
    conn.execute("INSTALL vss; LOAD vss;")
    conn.execute("SET hnsw_enable_experimental_persistence = true;")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH):
    """Create tables (including the embedding column) via SQLModel."""
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)
    engine.dispose()


def create_fts_index(db_path: Path = DEFAULT_DB_PATH):
    """Create a full-text search index on recipes.full_text."""
    conn = get_connection(db_path)
    conn.execute(
        "PRAGMA create_fts_index("
        "  'recipes', 'id', 'full_text',"
        "  stemmer='porter', overwrite=1"
        ")"
    )
    conn.close()


def create_vector_index(db_path: Path = DEFAULT_DB_PATH):
    """Create an HNSW vector index on recipes.embedding."""
    conn = get_connection(db_path)
    conn.execute("DROP INDEX IF EXISTS idx_recipes_embedding;")
    conn.execute(
        "CREATE INDEX idx_recipes_embedding ON recipes "
        "USING HNSW (embedding) WITH (metric = 'cosine')"
    )
    conn.close()
