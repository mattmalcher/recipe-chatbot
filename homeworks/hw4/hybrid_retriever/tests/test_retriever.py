"""Tests for the hybrid_retriever module."""
from matplotlib.pylab import isin

import pytest

from homeworks.hw4.hybrid_retriever.db import EMBEDDING_DIM, get_connection, init_db
from homeworks.hw4.hybrid_retriever.models import RecipeBase, SearchResult
from homeworks.hw4.hybrid_retriever.retriever import (
    HybridRetriever,
    _fts_search,
    _vector_search,
    reciprocal_rank_fusion,
)
from homeworks.hw4.hybrid_retriever.tests.conftest import _make_fake_embedding


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (pure logic, no DB)
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_ranking(self):
        ranking = [(1, 2.5), (2, 1.0)]
        fused = reciprocal_rank_fusion(ranking, k=60)

        assert fused[0][0] == 1
        assert fused[1][0] == 2
        assert fused[0][1] == pytest.approx(1.0 / (60 + 1))
        assert fused[1][1] == pytest.approx(1.0 / (60 + 2))

    def test_two_rankings_overlap_boosts(self):
        r1 = [(1, 2.5), (2, 1.0)]
        r2 = [(1, 3.0), (3, 0.5)]
        fused = reciprocal_rank_fusion(r1, r2, k=60)

        # Doc 1 appears in both lists so it should be ranked first
        assert fused[0][0] == 1
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert fused[0][1] == pytest.approx(expected)

    def test_two_rankings_disjoint(self):
        r1 = [(1, 2.5)]
        r2 = [(2, 3.0)]
        fused = reciprocal_rank_fusion(r1, r2, k=60)

        ids = {doc_id for doc_id, _ in fused}
        assert ids == {1, 2}
        # Both have equal score (rank 1 in their respective lists)
        assert fused[0][1] == fused[1][1]

    def test_custom_k(self):
        ranking = [(1, 2.5)]
        fused_low = reciprocal_rank_fusion(ranking, k=1)
        fused_high = reciprocal_rank_fusion(ranking, k=1000)

        assert fused_low[0][1] > fused_high[0][1]

    def test_preserves_order(self):
        r1 = [(1, 3.0), (2, 2.0), (3, 1.0)]
        fused = reciprocal_rank_fusion(r1, k=60)

        doc_ids = [doc_id for doc_id, _ in fused]
        assert doc_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Database init
# ---------------------------------------------------------------------------


class TestDatabase:
    def test_init_creates_file(self, db_path):
        init_db(db_path)
        assert db_path.exists()

    def test_has_embedding_column(self, db_path):
        init_db(db_path)
        conn = get_connection(db_path)
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'recipes'"
        ).fetchall()
        conn.close()

        col_map = {name: dtype for name, dtype in cols}
        assert "embedding" in col_map
        assert f"FLOAT[{EMBEDDING_DIM}]" in col_map["embedding"]

    def test_init_idempotent(self, db_path):
        init_db(db_path)
        init_db(db_path)  # should not raise
        conn = get_connection(db_path)
        count = conn.execute("SELECT count(*) FROM recipes").fetchone()[0]
        conn.close()
        assert count == 0


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


class TestFTSSearch:
    def test_returns_results(self, seeded_session):
        results = _fts_search(seeded_session, "chocolate cake", top_k=5)
        assert len(results) > 0
        assert results[0][0] == 1  # chocolate cake recipe id

    def test_ranking_order(self, seeded_session):
        results = _fts_search(seeded_session, "chicken pasta", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, seeded_session):
        results = _fts_search(seeded_session, "dinner", top_k=1)
        assert len(results) <= 1

    def test_no_match(self, seeded_session):
        results = _fts_search(seeded_session, "xyznonexistentterm", top_k=5)
        assert results == []

    def test_returns_tuples(self, seeded_session):
        results = _fts_search(seeded_session, "chocolate", top_k=1)
        assert len(results) > 0
        doc_id, score = results[0]
        assert isinstance(doc_id, int)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_returns_results(self, populated_session):
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(populated_session, query_emb, top_k=5)
        assert len(results) > 0

    def test_respects_top_k(self, populated_session):
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(populated_session, query_emb, top_k=1)
        assert len(results) == 1

    def test_score_range(self, populated_session):
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(populated_session, query_emb, top_k=5)
        for _doc_id, score in results:
            assert -1.0 <= score <= 1.0

    def test_returns_tuples(self, populated_session):
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(populated_session, query_emb, top_k=1)
        doc_id, score = results[0]
        assert isinstance(doc_id, int)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# HybridRetriever integration
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def test_retrieve_fts_mode(self, retriever):
        results = retriever.retrieve("chocolate", top_k=3, mode="fts")
        assert len(results) > 0
        assert results[0].name == "chocolate cake"

    def test_retrieve_vector_mode(self, retriever, monkeypatch):
        fake_emb = _make_fake_embedding(seed=1)  # same seed as recipe 1
        monkeypatch.setattr(
            "homeworks.hw4.hybrid_retriever.retriever.generate_query_embedding",
            lambda query, model: fake_emb,
        )
        results = retriever.retrieve("anything", top_k=3, mode="vector")
        assert len(results) > 0

    def test_retrieve_hybrid_mode(self, retriever, monkeypatch):
        fake_emb = _make_fake_embedding(seed=1)
        monkeypatch.setattr(
            "homeworks.hw4.hybrid_retriever.retriever.generate_query_embedding",
            lambda query, model: fake_emb,
        )
        results = retriever.retrieve("chocolate", top_k=3, mode="hybrid")
        assert len(results) > 0

    def test_retrieve_bm25_alias(self, retriever, monkeypatch):
        fake_emb = _make_fake_embedding(seed=1)
        monkeypatch.setattr(
            "homeworks.hw4.hybrid_retriever.retriever.generate_query_embedding",
            lambda query, model: fake_emb,
        )
        results = retriever.retrieve_bm25("chocolate", top_k=3)
        assert len(results) > 0

    def test_result_has_required_keys(self, retriever):
        results = retriever.retrieve("pasta", top_k=3, mode="fts")
        for r in results:
            assert isinstance(r, SearchResult)

    def test_result_includes_recipe_data(self, retriever):
        results: list[SearchResult] = retriever.retrieve("chocolate", top_k=1, mode="fts")
        r: SearchResult = results[0]
        # Recipe fields come from SQLModel, JSON columns are auto-deserialized
        assert isinstance(r.ingredients, list)
        assert r.minutes == 60
        assert r.name == "chocolate cake"


# ---------------------------------------------------------------------------
# RecipeBase serialization
# ---------------------------------------------------------------------------


class TestRecipeBaseSerialization:
    def test_to_embedding_text_full_recipe(self):
        """Test that to_embedding_text generates properly formatted output."""
        recipe = RecipeBase(
            id=1,
            name="Chocolate Chip Cookies",
            description="Classic homemade cookies",
            ingredients=["flour", "butter", "sugar", "eggs", "chocolate chips"],
            steps=[
                "Cream butter and sugar",
                "Add eggs",
                "Mix in flour",
                "Fold in chocolate chips",
                "Bake at 350F",
            ],
            tags=["dessert", "cookies", "chocolate"],
            minutes=30,
            n_ingredients=5,
            n_steps=5,
        )

        text = recipe.to_embedding_text()

        # Verify all sections present
        assert "Recipe: Chocolate Chip Cookies" in text
        assert "Description: Classic homemade cookies" in text
        assert "Ingredients:" in text
        assert "- flour" in text
        assert "- butter" in text
        assert "Steps:" in text
        assert "1. Cream butter and sugar" in text
        assert "2. Add eggs" in text
        assert "Tags: dessert, cookies, chocolate" in text
        assert "Cooking Time: 30 minutes" in text
        assert "Number of Ingredients: 5" in text
        assert "Number of Steps: 5" in text

    def test_to_embedding_text_minimal(self):
        """Test edge case with minimal fields."""
        recipe = RecipeBase(
            id=2,
            name="Simple Toast",
            description="",
            ingredients=None,
            steps=None,
            tags=None,
            minutes=0,
        )

        text = recipe.to_embedding_text()
        assert text == "# Recipe: Simple Toast"

    def test_to_embedding_text_no_description(self):
        """Test recipe without description."""
        recipe = RecipeBase(
            id=3,
            name="Quick Pasta",
            description="",
            ingredients=["pasta", "sauce"],
            steps=["Boil pasta", "Add sauce"],
            tags=["quick", "dinner"],
            minutes=15,
            n_ingredients=2,
            n_steps=2,
        )

        text = recipe.to_embedding_text()
        assert "Recipe: Quick Pasta" in text
        assert "Description:" not in text  # Should not have description section
        assert "Ingredients:" in text
        assert "- pasta" in text
        assert "Steps:" in text
        assert "1. Boil pasta" in text

    def test_to_embedding_text_empty_lists(self):
        """Test recipe with empty ingredient/step/tag lists."""
        recipe = RecipeBase(
            id=4,
            name="Mystery Recipe",
            description="A mysterious dish",
            ingredients=[],
            steps=[],
            tags=[],
            minutes=20,
            n_ingredients=0,
            n_steps=0,
        )

        text = recipe.to_embedding_text()
        assert "Recipe: Mystery Recipe" in text
        assert "Description: A mysterious dish" in text
        # Empty lists should not create sections
        assert "Ingredients:" not in text
        assert "Steps:" not in text
        assert "Tags:" not in text
        # Zero counts should not be displayed
        assert "Number of Ingredients:" not in text
        assert "Number of Steps:" not in text
        assert "Cooking Time: 20 minutes" in text

    def test_to_embedding_text_structure(self):
        """Test that the output has proper structure and formatting."""
        recipe = RecipeBase(
            id=5,
            name="Test Recipe",
            description="Test description",
            ingredients=["item1", "item2"],
            steps=["step1", "step2", "step3"],
            tags=["tag1", "tag2"],
            minutes=45,
            n_ingredients=2,
            n_steps=3,
        )

        text = recipe.to_embedding_text()

        # Check order: name comes first, then description, then ingredients, etc.
        name_pos = text.find("Recipe:")
        desc_pos = text.find("Description:")
        ing_pos = text.find("Ingredients:")
        steps_pos = text.find("Steps:")
        tags_pos = text.find("Tags:")

        assert name_pos < desc_pos < ing_pos < steps_pos < tags_pos

        # Check numbered steps
        assert "1. step1" in text
        assert "2. step2" in text
        assert "3. step3" in text

        # Check bulleted ingredients
        assert "- item1" in text
        assert "- item2" in text
