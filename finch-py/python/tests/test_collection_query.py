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


class CollectionQueryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="finch_test_query_")
        self.path = f"{self._tmp.name}/test_collection"
        self.schema = _make_schema()
        self.option = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        self.coll = finch_db.create_and_open(path=self.path, schema=self.schema, option=self.option)

        self.docs = [
            finch_db.Doc(
                id=f"{i}",
                fields={"id": i, "name": "test", "weight": 80.0, "height": 210},
                vectors={"dense": [i + 0.1] * 16, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
            )
            for i in range(1, 101)
        ]
        statuses = self.coll.insert(self.docs)
        self.assertEqual(len(statuses), len(self.docs))
        for s in statuses:
            self.assertTrue(s.ok())

    def tearDown(self):
        try:
            self.coll.destroy()
        finally:
            self._tmp.cleanup()

    def test_query_default_topk(self):
        out = self.coll.query()
        self.assertEqual(len(out), 10)

        out2 = self.coll.query(topk=5)
        self.assertEqual(len(out2), 5)

    def test_query_include_vector(self):
        out = self.coll.query(topk=1, include_vector=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].vector_names(), [])

        out2 = self.coll.query(topk=1, include_vector=True)
        self.assertEqual(len(out2), 1)
        self.assertIn("dense", out2[0].vector_names())
        self.assertIn("sparse", out2[0].vector_names())

    def test_query_output_fields(self):
        out = self.coll.query(topk=1, output_fields=["id", "name"])
        self.assertEqual(len(out), 1)
        doc = out[0]
        self.assertEqual(set(doc.field_names()), {"id", "name"})

    def test_query_range_and_in_filters(self):
        idx = self.docs[10].id  # string of digits

        out = self.coll.query(filter=f"id>{idx}", topk=1000)
        self.assertEqual(len(out), len(self.docs) - 11)

        out = self.coll.query(filter=f"id>={idx}", topk=1000)
        self.assertEqual(len(out), len(self.docs) - 10)

        out = self.coll.query(filter=f"id<{idx}", topk=1000)
        self.assertEqual(len(out), 10)

        out = self.coll.query(filter=f"id<={idx}", topk=1000)
        self.assertEqual(len(out), 11)

        out = self.coll.query(filter=f"id={idx}", topk=1000)
        self.assertEqual(len(out), 1)

        out = self.coll.query(filter=f"id!={idx}", topk=1000)
        self.assertEqual(len(out), len(self.docs) - 1)

        l_id, r_id = self.docs[10].id, self.docs[90].id
        out = self.coll.query(filter=f"id>{l_id} and id<{r_id}", topk=1000)
        self.assertEqual(len(out), 79)

        out = self.coll.query(filter=f"id in (1)", topk=1000)
        self.assertEqual(len(out), 1)

        out = self.coll.query(filter="id not in (1)", topk=1000)
        self.assertEqual(len(out), len(self.docs) - 1)

    def test_query_invalid_vector_query_inputs(self):
        with self.assertRaises(ValueError):
            self.coll.query(
                finch_db.VectorQuery(
                    field_name="dense",
                    id=self.docs[0].id,
                    vector=self.docs[0].vector("dense"),
                )
            )

        bad = finch_db.VectorQuery(
            field_name="dense",
            vector=self.docs[0].vector("dense"),
            param=[1, 2, 3],  # type: ignore[arg-type]
        )
        with self.assertRaises(TypeError):
            self.coll.query(vectors=bad, filter="id in (1)", topk=100)

        with self.assertRaises(ValueError):
            self.coll.query(
                [
                    finch_db.VectorQuery(field_name="dense", vector=self.docs[0].vector("dense")),
                    finch_db.VectorQuery(field_name="dense", vector=self.docs[0].vector("dense")),
                ]
            )

    def test_delete_by_filter(self):
        self.coll.delete_by_filter(filter="id=1")
        self.assertEqual(self.coll.stats.doc_count, 99)

