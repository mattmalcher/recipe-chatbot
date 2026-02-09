"""Tests for the hybrid_retriever module."""

import pytest

from homeworks.hw4.hybrid_retriever.db import EMBEDDING_DIM, get_connection, init_db
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
        ranking = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ]
        fused = reciprocal_rank_fusion(ranking, k=60)

        assert fused[0]["id"] == 1
        assert fused[1]["id"] == 2
        assert fused[0]["rrf_score"] == pytest.approx(1.0 / (60 + 1))
        assert fused[1]["rrf_score"] == pytest.approx(1.0 / (60 + 2))

    def test_two_rankings_overlap_boosts(self):
        r1 = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        r2 = [{"id": 1, "name": "a"}, {"id": 3, "name": "c"}]
        fused = reciprocal_rank_fusion(r1, r2, k=60)

        # Doc 1 appears in both lists so it should be ranked first
        assert fused[0]["id"] == 1
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert fused[0]["rrf_score"] == pytest.approx(expected)

    def test_two_rankings_disjoint(self):
        r1 = [{"id": 1, "name": "a"}]
        r2 = [{"id": 2, "name": "b"}]
        fused = reciprocal_rank_fusion(r1, r2, k=60)

        ids = {item["id"] for item in fused}
        assert ids == {1, 2}
        # Both have equal score (rank 1 in their respective lists)
        assert fused[0]["rrf_score"] == fused[1]["rrf_score"]

    def test_custom_k(self):
        ranking = [{"id": 1, "name": "a"}]
        fused_low = reciprocal_rank_fusion(ranking, k=1)
        fused_high = reciprocal_rank_fusion(ranking, k=1000)

        assert fused_low[0]["rrf_score"] > fused_high[0]["rrf_score"]

    def test_hybrid_rank_assigned(self):
        r1 = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
        fused = reciprocal_rank_fusion(r1, k=60)

        for i, item in enumerate(fused):
            assert item["hybrid_rank"] == i + 1


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
    def test_returns_results(self, seeded_db):
        conn = get_connection(seeded_db)
        results = _fts_search(conn, "chocolate cake", top_k=5)
        conn.close()

        assert len(results) > 0
        assert results[0]["id"] == 1  # chocolate cake recipe

    def test_ranking_order(self, seeded_db):
        conn = get_connection(seeded_db)
        results = _fts_search(conn, "chicken pasta", top_k=5)
        conn.close()

        scores = [r["fts_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, seeded_db):
        conn = get_connection(seeded_db)
        results = _fts_search(conn, "dinner", top_k=1)
        conn.close()

        assert len(results) <= 1

    def test_no_match(self, seeded_db):
        conn = get_connection(seeded_db)
        results = _fts_search(conn, "xyznonexistentterm", top_k=5)
        conn.close()

        assert results == []

    def test_result_keys(self, seeded_db):
        conn = get_connection(seeded_db)
        results = _fts_search(conn, "chocolate", top_k=1)
        conn.close()

        assert len(results) > 0
        r = results[0]
        assert set(r.keys()) == {"id", "name", "fts_score", "fts_rank"}


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_returns_results(self, populated_db):
        conn = get_connection(populated_db)
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(conn, query_emb, top_k=5)
        conn.close()

        assert len(results) > 0

    def test_respects_top_k(self, populated_db):
        conn = get_connection(populated_db)
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(conn, query_emb, top_k=1)
        conn.close()

        assert len(results) == 1

    def test_score_range(self, populated_db):
        conn = get_connection(populated_db)
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(conn, query_emb, top_k=5)
        conn.close()

        for r in results:
            assert -1.0 <= r["vector_score"] <= 1.0

    def test_result_keys(self, populated_db):
        conn = get_connection(populated_db)
        query_emb = _make_fake_embedding(seed=999)
        results = _vector_search(conn, query_emb, top_k=1)
        conn.close()

        r = results[0]
        assert set(r.keys()) == {"id", "name", "vector_distance", "vector_score", "vector_rank"}


# ---------------------------------------------------------------------------
# HybridRetriever integration
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def test_retrieve_fts_mode(self, retriever):
        results = retriever.retrieve("chocolate", top_k=3, mode="fts")
        assert len(results) > 0
        assert results[0]["name"] == "chocolate cake"

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
            assert "id" in r
            assert "name" in r
            assert "bm25_score" in r
            assert "rank" in r

    def test_enrich_merges_recipe_data(self, retriever):
        results = retriever.retrieve("chocolate", top_k=1, mode="fts")
        r = results[0]
        # These fields come from the recipe dict, not from FTS
        assert "ingredients" in r
        assert "minutes" in r
        assert r["minutes"] == 60
