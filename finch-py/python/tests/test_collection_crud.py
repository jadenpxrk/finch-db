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


class CollectionCrudTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="finch_test_crud_")
        self.path = f"{self._tmp.name}/test_collection"
        self.schema = _make_schema()
        self.option = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        self.coll = finch_db.create_and_open(path=self.path, schema=self.schema, option=self.option)

    def tearDown(self):
        try:
            self.coll.destroy()
        finally:
            self._tmp.cleanup()

    def test_stats_index_completeness(self):
        stats = self.coll.stats
        self.assertEqual(stats.doc_count, 0)
        self.assertIn("dense", stats.index_completeness)
        self.assertIn("sparse", stats.index_completeness)
        self.assertEqual(stats.index_completeness["dense"], 1)
        self.assertEqual(stats.index_completeness["sparse"], 1)

    def test_create_and_drop_invert_index(self):
        before = self.coll.schema.field("weight")
        self.assertIsNotNone(before)
        self.assertIsNone(before.index_param)

        self.coll.create_index("weight", finch_db.InvertIndexParam())
        after = self.coll.schema.field("weight")
        self.assertIsNotNone(after)
        self.assertIsNotNone(after.index_param)
        self.assertEqual(after.index_param.type, finch_db.IndexType.INVERT)
        self.assertFalse(after.index_param.enable_range_optimization)

        self.coll.drop_index("weight")
        dropped = self.coll.schema.field("weight")
        self.assertIsNotNone(dropped)
        self.assertIsNone(dropped.index_param)

    def test_insert_fetch_roundtrip(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 0, "name": "test", "weight": 80.0, "height": 140},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0, 2: 2.0}},
        )
        s = self.coll.insert(doc)
        self.assertTrue(s.ok())
        self.assertEqual(self.coll.stats.doc_count, 1)

        out = self.coll.fetch(doc.id)
        self.assertIn(doc.id, out)
        got = out[doc.id]
        self.assertEqual(got.field("id"), 0)
        self.assertEqual(got.field("name"), "test")
        self.assertEqual(got.field("weight"), 80.0)
        self.assertEqual(got.field("height"), 140)

    def test_insert_missing_non_nullable_field_raises(self):
        doc = finch_db.Doc(id="0", vectors={"dense": [0.1] * 16})
        with self.assertRaises(ValueError) as ctx:
            self.coll.insert(doc)
        self.assertIn("field[id] is configured not nullable", str(ctx.exception))

        doc2 = finch_db.Doc(id="0", fields={"id": 1}, vectors={"dense": [0.1] * 16})
        with self.assertRaises(ValueError) as ctx2:
            self.coll.insert(doc2)
        self.assertIn("field[name] is configured not nullable", str(ctx2.exception))

    def test_insert_nullable_true_fields_default_to_none(self):
        doc = finch_db.Doc(
            id="0",
            fields={"id": 1, "name": "test"},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        s = self.coll.insert(doc)
        self.assertTrue(s.ok())

        out = self.coll.fetch([doc.id])
        got = out[doc.id]
        self.assertEqual(got.field("id"), 1)
        self.assertEqual(got.field("name"), "test")
        self.assertIsNone(got.field("weight"))
        self.assertIsNone(got.field("height"))

    def test_update_non_nullable_field_set_to_none_raises(self):
        base = finch_db.Doc(
            id="0",
            fields={"id": 1, "name": "test", "weight": 80.0, "height": 140},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        self.assertTrue(self.coll.insert(base).ok())

        # Setting a non-nullable field to None should error.
        with self.assertRaises(ValueError) as ctx:
            self.coll.update(finch_db.Doc(id="0", fields={"id": None}))
        self.assertIn("configured not nullable", str(ctx.exception))

        # Setting a nullable field to None is allowed.
        s = self.coll.update(finch_db.Doc(id="0", fields={"weight": None}))
        self.assertTrue(s.ok())
        got = self.coll.fetch("0")["0"]
        self.assertIsNone(got.field("weight"))
