from __future__ import annotations

import math
import os
import unittest
from unittest.mock import MagicMock, patch

import finch_db


class RerankerTest(unittest.TestCase):
    def test_rrf_reranker(self):
        reranker = finch_db.RrfReRanker(topn=5, rerank_field="content", rank_constant=100)
        self.assertEqual(reranker.topn, 5)
        self.assertEqual(reranker.rerank_field, "content")
        self.assertEqual(reranker.rank_constant, 100)

        reranker2 = finch_db.RrfReRanker(rank_constant=60)
        self.assertEqual(reranker2._rrf_score(0), 1.0 / (60 + 0 + 1))
        self.assertEqual(reranker2._rrf_score(1), 1.0 / (60 + 1 + 1))

        doc1 = finch_db.Doc(id="1", score=0.8)
        doc2 = finch_db.Doc(id="2", score=0.7)
        doc3 = finch_db.Doc(id="3", score=0.9)
        doc4 = finch_db.Doc(id="4", score=0.6)
        results = reranker2.rerank({"v1": [doc1, doc2, doc3], "v2": [doc3, doc1, doc4]})
        scores = [d.score for d in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_weighted_reranker(self):
        weights = {"vector1": 0.7, "vector2": 0.3}
        reranker = finch_db.WeightedReRanker(
            topn=5,
            rerank_field="content",
            metric=finch_db.MetricType.L2,
            weights=weights,
        )
        self.assertEqual(reranker.topn, 5)
        self.assertEqual(reranker.rerank_field, "content")
        self.assertEqual(reranker.metric, finch_db.MetricType.L2)
        self.assertEqual(reranker.weights, weights)

        r = finch_db.WeightedReRanker()
        self.assertEqual(r._normalize_score(1.0, finch_db.MetricType.L2), 1.0 - 2 * math.atan(1.0) / math.pi)
        self.assertEqual(r._normalize_score(1.0, finch_db.MetricType.IP), 0.5 + math.atan(1.0) / math.pi)
        self.assertEqual(r._normalize_score(1.0, finch_db.MetricType.COSINE), 1.0 - 1.0 / 2.0)
        with self.assertRaises(ValueError):
            r._normalize_score(1.0, "unsupported_metric")  # type: ignore[arg-type]

        doc1 = finch_db.Doc(id="1", score=0.8)
        doc2 = finch_db.Doc(id="2", score=0.7)
        doc3 = finch_db.Doc(id="3", score=0.9)
        out = reranker.rerank({"vector1": [doc1, doc2], "vector2": [doc2, doc3]})
        scores = [d.score for d in out]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_qwen_reranker_init_and_rerank(self):
        with self.assertRaisesRegex(ValueError, "Query is required for QwenReRanker"):
            finch_db.QwenReRanker(api_key="test_key")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DashScope API key is required"):
                finch_db.QwenReRanker(query="test", rerank_field="content")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            rr = finch_db.QwenReRanker(query="test", rerank_field="content")
            self.assertEqual(rr.query, "test")
            self.assertEqual(rr._api_key, "test_key")

        rr2 = finch_db.QwenReRanker(query="test", api_key="explicit_key", rerank_field="content")
        self.assertEqual(rr2._api_key, "explicit_key")

        rr3 = finch_db.QwenReRanker(query="test", api_key="test_key", rerank_field="content")
        self.assertEqual(rr3.model, "gte-rerank-v2")
        rr4 = finch_db.QwenReRanker(query="test", model="custom-model", api_key="test_key", rerank_field="content")
        self.assertEqual(rr4.model, "custom-model")

        self.assertEqual(rr3.rerank({}), [])

        with self.assertRaisesRegex(ValueError, "No documents to rerank"):
            rr3.rerank({"vector1": [finch_db.Doc(id="1")]})

        with self.assertRaisesRegex(ValueError, "No documents to rerank"):
            rr3.rerank({"vector1": [finch_db.Doc(id="1", fields={"content": ""}), finch_db.Doc(id="2", fields={"content": "   "})]})

        with patch("finch_db.extension.qwen_function.require_module") as mock_require_module:
            mock_dashscope = MagicMock()
            mock_require_module.return_value = mock_dashscope

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.output = {
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.85},
                ]
            }
            mock_dashscope.TextReRank.call.return_value = mock_response

            rr = finch_db.QwenReRanker(query="test query", topn=2, api_key="test_key", rerank_field="content")
            query_results = {
                "vector1": [
                    finch_db.Doc(id="1", fields={"content": "Document 1"}),
                    finch_db.Doc(id="2", fields={"content": "Document 2"}),
                ]
            }
            results = rr.rerank(query_results)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].id, "1")
            self.assertEqual(results[0].score, 0.95)
            self.assertEqual(results[1].id, "2")
            self.assertEqual(results[1].score, 0.85)

            mock_dashscope.TextReRank.call.assert_called_once_with(
                model="gte-rerank-v2",
                query="test query",
                documents=["Document 1", "Document 2"],
                top_n=2,
                return_documents=False,
            )

    def test_default_local_reranker_mocked(self):
        with self.assertRaisesRegex(ValueError, "Query is required for DefaultLocalReRanker"):
            finch_db.DefaultLocalReRanker(rerank_field="content")

        with self.assertRaisesRegex(ValueError, "Query is required for DefaultLocalReRanker"):
            finch_db.DefaultLocalReRanker(query="", rerank_field="content")

        with patch("finch_db.extension.sentence_transformer_rerank_function.require_module") as mock_require_module:
            mock_st = MagicMock()
            mock_model = MagicMock()
            mock_model.predict = MagicMock()
            mock_model.device = "cpu"
            mock_st.CrossEncoder.return_value = mock_model
            mock_require_module.return_value = mock_st

            rr = finch_db.DefaultLocalReRanker(
                query="test query",
                topn=5,
                rerank_field="content",
                model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            )
            self.assertEqual(rr.query, "test query")
            self.assertEqual(rr.topn, 5)
            self.assertEqual(rr.rerank_field, "content")
            self.assertEqual(rr.model_name, "cross-encoder/ms-marco-MiniLM-L6-v2")
            self.assertEqual(rr.model_source, "huggingface")
            self.assertEqual(rr.batch_size, 32)

            self.assertEqual(rr.rerank({}), [])

            with self.assertRaisesRegex(ValueError, "No documents to rerank"):
                rr.rerank({"vector1": [finch_db.Doc(id="1")]})

            query_results = {"vector1": [finch_db.Doc(id="1", fields={"content": "Document 1"}), finch_db.Doc(id="2", fields={"content": "Document 2"})]}
            mock_model.predict.return_value = [0.9, 0.8]
            out = rr.rerank(query_results)
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0].id, "1")
            self.assertTrue(out[0].score > out[1].score)

