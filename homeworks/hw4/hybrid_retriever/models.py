"""SQLModel table definitions for recipe storage in DuckDB."""

from typing import Optional

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


class Recipe(SQLModel, table=True):
    """Recipe table stored in DuckDB.

    The embedding column (FLOAT[3072]) is added via raw SQL in db.py
    because SQLModel cannot represent DuckDB's fixed-size array type.
    """

    __tablename__ = "recipes"

    id: int = Field(
        sa_column=sa.Column(sa.Integer, primary_key=True, autoincrement=False)
    )
    name: str
    description: str = Field(default="", sa_column=sa.Column(sa.Text))
    minutes: int = Field(default=0)
    n_ingredients: int = Field(default=0)
    n_steps: int = Field(default=0)
    submitted: Optional[str] = Field(default=None)
    contributor_id: Optional[int] = Field(default=None)
    full_text: str = Field(default="", sa_column=sa.Column(sa.Text))
    ingredients_text: str = Field(default="", sa_column=sa.Column(sa.Text))
    steps_text: str = Field(default="", sa_column=sa.Column(sa.Text))
    tags_text: str = Field(default="", sa_column=sa.Column(sa.Text))
