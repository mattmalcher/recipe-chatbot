"""Tests for the hybrid_retriever module."""
from matplotlib.pylab import isin

import pytest

from homeworks.hw4.hybrid_retriever.db import EMBEDDING_DIM, get_connection, init_db
from homeworks.hw4.hybrid_retriever.models import SearchResult
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
