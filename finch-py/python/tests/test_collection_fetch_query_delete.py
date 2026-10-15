from __future__ import annotations

import tempfile
import unittest

import finch_db


def _make_schema() -> finch_db.CollectionSchema:
    return finch_db.CollectionSchema(
        name="test_collection",
        fields=[
            finch_db.FieldSchema(
                "id",
                finch_db.DataType.INT64,
                nullable=False,
                index_param=finch_db.InvertIndexParam(enable_range_optimization=True),
            ),
            finch_db.FieldSchema(
                "name",
                finch_db.DataType.STRING,
                nullable=False,
                index_param=finch_db.InvertIndexParam(),
            ),
            finch_db.FieldSchema("weight", finch_db.DataType.FLOAT, nullable=True),
            finch_db.FieldSchema("height", finch_db.DataType.INT32, nullable=True),
        ],
        vectors=[
            finch_db.VectorSchema(
                "dense",
                data_type=finch_db.DataType.VECTOR_FP32,
                dimension=16,
                index_param=finch_db.HnswIndexParam(),
            ),
            finch_db.VectorSchema(
                "sparse",
                data_type=finch_db.DataType.SPARSE_VECTOR_FP32,
                dimension=0,
                index_param=finch_db.HnswIndexParam(),
            ),
        ],
    )


class CollectionFetchQueryDeleteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="finch_test_collection_more_")
        self.path = f"{self._tmp.name}/test_collection"
        self.schema = _make_schema()
        self.option = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        self.coll = finch_db.create_and_open(path=self.path, schema=self.schema, option=self.option)

    def tearDown(self):
        try:
            self.coll.destroy()
        finally:
            self._tmp.cleanup()

    def test_init_api_exists(self):
        finch_db.init(log_type=finch_db.LogType.CONSOLE, log_level=finch_db.LogLevel.ERROR, log_dir="./log")

    def test_add_column_nullable_false_rejected(self):
        with self.assertRaises(ValueError):
            self.coll.add_column(finch_db.FieldSchema("age", finch_db.DataType.INT32, nullable=False))

    def test_fetch_skips_missing_ids(self):
        docs = [
            finch_db.Doc(
                id=f"{i}",
                fields={"id": i, "name": "t"},
                vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
            )
            for i in range(1, 6)
        ]
        self.coll.insert(docs)
        out = self.coll.fetch(["x"] + [d.id for d in docs])
        self.assertEqual(len(out), 5)
        self.assertNotIn("x", out)

    def test_query_empty_collection_returns_empty(self):
        self.assertEqual(self.coll.stats.doc_count, 0)
        out = self.coll.query()
        self.assertEqual(len(out), 0)

    def test_query_by_id_vector(self):
        docs = [
            finch_db.Doc(
                id=f"{i}",
                fields={"id": i, "name": "t"},
                vectors={"dense": [i + 0.1] * 16, "sparse": {1: 1.0}},
            )
            for i in range(1, 31)
        ]
        self.coll.insert(docs)
        out = self.coll.query(finch_db.VectorQuery(field_name="dense", id=docs[0].id))
        self.assertEqual(len(out), 10)

    def test_delete_existing_doc(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "t"},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        self.assertTrue(self.coll.insert(doc).ok())
        self.assertEqual(self.coll.stats.doc_count, 1)
        s = self.coll.delete(doc.id)
        self.assertTrue(s.ok())
        self.assertEqual(self.coll.stats.doc_count, 0)

    def test_optimize_does_not_throw(self):
        self.coll.optimize(option=finch_db.OptimizeOption())

