"""SQLModel table definitions for recipe storage in DuckDB."""
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlmodel import Field, SQLModel

EMBEDDING_DIM = 3072


class DuckDBArrayType(sa.types.TypeEngine):
    """Custom SQLAlchemy type for DuckDB fixed-size arrays (e.g. FLOAT[3072]).

    DuckDB stores vector embeddings as fixed-size typed arrays like FLOAT[N],
    but neither SQLAlchemy nor SQLModel has a built-in type for this.
    duckdb_engine inherits from the PostgreSQL dialect, which only supports
    variable-length ARRAY — not the fixed-size variant DuckDB uses for VSS.

    This type uses SQLAlchemy's @compiles extension (the same mechanism
    duckdb_engine itself uses for Struct/Map in datatypes.py) to emit the
    correct DDL so that SQLModel.metadata.create_all() can create the
    column without falling back to raw SQL ALTER TABLE.
    """

    __visit_name__ = "duckdb_array"
    cache_ok = True

    def __init__(self, item_type: str = "FLOAT", size: int = EMBEDDING_DIM):
        self.item_type = item_type
        self.size = size


@compiles(DuckDBArrayType, "duckdb")
def _compile_duckdb_array(type_, compiler, **kw):
    return f"{type_.item_type}[{type_.size}]"

class RecipeBase(SQLModel, table=False):
    id: int
    name: str
    description: str = Field(default="", sa_column=sa.Column(sa.Text))
    minutes: int = Field(default=0)
    n_ingredients: int = Field(default=0)
    n_steps: int = Field(default=0)
    submitted: Optional[str] = Field(default=None)
    contributor_id: Optional[int] = Field(default=None)
    full_text: str = Field(default="", sa_column=sa.Column(sa.Text))
    ingredients: Optional[list] = Field(default=None, sa_column=sa.Column(sa.JSON))
    steps: Optional[list] = Field(default=None, sa_column=sa.Column(sa.JSON))
    tags: Optional[list] = Field(default=None, sa_column=sa.Column(sa.JSON))
    nutrition: Optional[dict] = Field(default=None, sa_column=sa.Column(sa.JSON))

    def to_embedding_text(self) -> str:
        """Generate a structured text representation for embedding generation.

        Creates a human-readable, labeled format that includes recipe name,
        description, ingredients, steps, tags, and relevant metadata.

        Returns:
            Formatted string suitable for embedding generation.
        """
        parts = []

        # Recipe name (most important, comes first)
        if self.name:
            parts.append(f"# Recipe:\n{self.name}")

        # Description
        if self.description:
            parts.append(f"\n\n# Description:\n{self.description}")

        # Ingredients (bulleted list)
        if self.ingredients:
            parts.append("\n\n# Ingredients:")
            for ingredient in self.ingredients:
                parts.append(f"\n- {ingredient}")

        # Steps (numbered list)
        if self.steps:
            parts.append("\n\n# Steps:")
            for i, step in enumerate(self.steps, start=1):
                parts.append(f"\n{i}. {step}")

        # Tags (comma-separated)
        if self.tags:
            tags_str = ", ".join(self.tags)
            parts.append(f"\n\n# Tags: {tags_str}")

        # Metadata (useful for filtering/context)
        metadata_parts = []
        if self.minutes and self.minutes > 0:
            metadata_parts.append(f"Cooking Time: {self.minutes} minutes")
        if self.n_ingredients and self.n_ingredients > 0:
            metadata_parts.append(f"Number of Ingredients: {self.n_ingredients}")
        if self.n_steps and self.n_steps > 0:
            metadata_parts.append(f"Number of Steps: {self.n_steps}")

        if metadata_parts:
            parts.append("\n\n# Metadata\n" + " | ".join(metadata_parts))

        return "".join(parts).strip()


class Recipe(RecipeBase, table=True):
    """Recipe table stored in DuckDB."""

    __tablename__ = "recipes"
    __table_args__ = {"extend_existing": True}

    id: int = Field(
        sa_column=sa.Column(sa.Integer, primary_key=True, autoincrement=False)
    )
    embedding: Optional[bytes] = Field(
        default=None,
        sa_column=sa.Column(DuckDBArrayType(), nullable=True),
    )
    

class SearchResult(RecipeBase, table=False):
    rank:int
    bm25_score:float
    