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


class CollectionDmlTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="finch_test_dml_")
        self.path = f"{self._tmp.name}/test_collection"
        self.schema = _make_schema()
        self.option = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        self.coll = finch_db.create_and_open(path=self.path, schema=self.schema, option=self.option)

    def tearDown(self):
        try:
            self.coll.destroy()
        finally:
            self._tmp.cleanup()

    def test_insert_duplicate_returns_status(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test", "weight": 80.0, "height": 140},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0, 2: 2.0}},
        )
        s1 = self.coll.insert(doc)
        self.assertTrue(s1.ok())
        self.assertEqual(self.coll.stats.doc_count, 1)

        s2 = self.coll.insert(doc)
        self.assertFalse(s2.ok())
        self.assertEqual(s2.code(), finch_db.StatusCode.ALREADY_EXISTS)
        self.assertEqual(self.coll.stats.doc_count, 1)

    def test_update_not_found_returns_status(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test"},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        s = self.coll.update(doc)
        self.assertFalse(s.ok())
        self.assertEqual(s.code(), finch_db.StatusCode.NOT_FOUND)
        self.assertEqual(self.coll.stats.doc_count, 0)

    def test_upsert_inserts_on_empty(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test"},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        s = self.coll.upsert(doc)
        self.assertTrue(s.ok())
        self.assertEqual(self.coll.stats.doc_count, 1)

    def test_upsert_existing_updates_and_keeps_doc_count(self):
        base = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test", "weight": 80.0, "height": 140},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        self.assertTrue(self.coll.insert(base).ok())
        self.assertEqual(self.coll.stats.doc_count, 1)

        updated = finch_db.Doc(id="0", fields={"weight": 81.0})
        s = self.coll.upsert(updated)
        self.assertTrue(s.ok())
        self.assertEqual(self.coll.stats.doc_count, 1)

        got = self.coll.fetch("0")["0"]
        self.assertEqual(got.field("id"), 0)
        self.assertEqual(got.field("name"), "test")
        self.assertAlmostEqual(float(got.field("weight")), 81.0, places=6)
        self.assertEqual(got.field("height"), 140)

    def test_delete_not_found_returns_status(self):
        s = self.coll.delete("does_not_exist")
        self.assertFalse(s.ok())
        self.assertEqual(s.code(), finch_db.StatusCode.NOT_FOUND)

    def test_delete_batch_not_found_returns_statuses(self):
        docs = [finch_db.Doc(id=f"{i}", fields={"id": i, "name": "t"}, vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}}) for i in range(3)]
        ids = [d.id for d in docs]
        res = self.coll.delete(ids)
        self.assertEqual(len(res), len(ids))
        for s in res:
            self.assertEqual(s.code(), finch_db.StatusCode.NOT_FOUND)

    def test_update_batch_not_found(self):
        docs = [
            finch_db.Doc(
                id=f"{i}",
                fields={"id": i, "name": "t"},
                vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
            )
            for i in range(3)
        ]
        res = self.coll.update(docs)
        self.assertEqual(len(res), 3)
        for s in res:
            self.assertEqual(s.code(), finch_db.StatusCode.NOT_FOUND)

    def test_update_existing_partial_fields(self):
        base = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test", "weight": 80.0, "height": 140},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        self.assertTrue(self.coll.insert(base).ok())

        s = self.coll.update(finch_db.Doc(id="0", fields={"id": 1}))
        self.assertTrue(s.ok())
        got = self.coll.fetch("0")["0"]
        self.assertEqual(got.field("id"), 1)
        self.assertEqual(got.field("name"), "test")
        self.assertAlmostEqual(float(got.field("weight")), 80.0, places=6)
        self.assertEqual(got.field("height"), 140)

        s2 = self.coll.update(finch_db.Doc(id="0", fields={"weight": None}))
        self.assertTrue(s2.ok())
        got2 = self.coll.fetch("0")["0"]
        self.assertIsNone(got2.field("weight"))

    def test_delete_by_filter_on_non_invert_field(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test", "weight": 80.0, "height": 140},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        self.assertTrue(self.coll.insert(doc).ok())
        self.assertEqual(self.coll.stats.doc_count, 1)

        # height is not invert-indexed in this schema; delete_by_filter should still work.
        self.coll.delete_by_filter(filter="height=140")
        self.assertEqual(self.coll.stats.doc_count, 0)
