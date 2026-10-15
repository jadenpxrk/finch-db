from __future__ import annotations

import math
import unittest

import finch_db


class DocConvertTest(unittest.TestCase):
    def test_py_doc_vectors_and_numpy(self):
        doc = finch_db.Doc(id="1", vectors={"dense": [1, 2, 3]})
        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.vector("dense"), [1, 2, 3])

        doc2 = finch_db.Doc(
            id="1",
            vectors={"dense": [1, 2, 3], "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        self.assertEqual(doc2.vector("dense"), [1, 2, 3])
        self.assertEqual(doc2.vector("sparse"), {1: 1.0, 2: 2.0, 3: 3.0})

        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        doc3 = finch_db.Doc._from_tuple(
            (
                "1",
                0.0,
                None,
                {"image": np.array([1, 2, 3]), "keys": {1: 1.0, 2: 2.0}},
            )
        )
        self.assertEqual(doc3.vector("image"), [1, 2, 3])
        self.assertEqual(doc3.vector("keys"), {1: 1.0, 2: 2.0})

    def test_core_doc_set_pk_score(self):
        from finch_db._finch import PyDoc as _Doc

        d = _Doc()
        d.set_pk("1")
        self.assertEqual(d.pk(), "1")
        d.set_score(0.9)
        self.assertTrue(math.isclose(d.score(), 0.9, rel_tol=1e-6))

    def test_core_doc_set_any_scalar_and_array(self):
        from finch_db._finch import PyDoc as _Doc

        d = _Doc()

        s = finch_db.FieldSchema("author", finch_db.DataType.STRING, nullable=True)
        d.set_any("author", s._get_object(), None)
        self.assertTrue(d.has_field("author"))
        self.assertIsNone(d.get_any("author", finch_db.DataType.STRING))

        s2 = finch_db.FieldSchema("author2", finch_db.DataType.STRING, nullable=False)
        with self.assertRaises(ValueError):
            d.set_any("author2", s2._get_object(), None)

        d.set_any("is_male", finch_db.FieldSchema("is_male", finch_db.DataType.BOOL)._get_object(), True)
        self.assertEqual(d.get_any("is_male", finch_db.DataType.BOOL), True)

        d.set_any("age", finch_db.FieldSchema("age", finch_db.DataType.INT32)._get_object(), 19)
        self.assertEqual(d.get_any("age", finch_db.DataType.INT32), 19)

        d.set_any("id", finch_db.FieldSchema("id", finch_db.DataType.INT64)._get_object(), 1111111111111111111)
        self.assertEqual(d.get_any("id", finch_db.DataType.INT64), 1111111111111111111)

        d.set_any("weight", finch_db.FieldSchema("weight", finch_db.DataType.FLOAT)._get_object(), 60.5)
        self.assertTrue(math.isclose(d.get_any("weight", finch_db.DataType.FLOAT), 60.5, rel_tol=1e-6))

        d.set_any("height", finch_db.FieldSchema("height", finch_db.DataType.DOUBLE)._get_object(), 1.77777777777)
        self.assertTrue(math.isclose(d.get_any("height", finch_db.DataType.DOUBLE), 1.77777777777, rel_tol=1e-9))

        d.set_any("u32", finch_db.FieldSchema("u32", finch_db.DataType.UINT32)._get_object(), 4294967295)
        self.assertEqual(d.get_any("u32", finch_db.DataType.UINT32), 4294967295)

        d.set_any("u64", finch_db.FieldSchema("u64", finch_db.DataType.UINT64)._get_object(), 18446744073709551615)
        self.assertEqual(d.get_any("u64", finch_db.DataType.UINT64), 18446744073709551615)

        d.set_any("tags", finch_db.FieldSchema("tags", finch_db.DataType.ARRAY_STRING)._get_object(), ["tag1", "tag2"])
        self.assertEqual(d.get_any("tags", finch_db.DataType.ARRAY_STRING), ["tag1", "tag2"])

        d.set_any("ids", finch_db.FieldSchema("ids", finch_db.DataType.ARRAY_INT32)._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("ids", finch_db.DataType.ARRAY_INT32), [1, 2, 3])

        d.set_any("ids64", finch_db.FieldSchema("ids64", finch_db.DataType.ARRAY_INT64)._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("ids64", finch_db.DataType.ARRAY_INT64), [1, 2, 3])

        d.set_any("weights", finch_db.FieldSchema("weights", finch_db.DataType.ARRAY_FLOAT)._get_object(), [1.0, 2.0])
        self.assertEqual(d.get_any("weights", finch_db.DataType.ARRAY_FLOAT), [1.0, 2.0])

        d.set_any("heights", finch_db.FieldSchema("heights", finch_db.DataType.ARRAY_DOUBLE)._get_object(), [1.0, 2.0, 3.0])
        self.assertEqual(d.get_any("heights", finch_db.DataType.ARRAY_DOUBLE), [1.0, 2.0, 3.0])

    def test_core_doc_set_any_vectors(self):
        from finch_db._finch import PyDoc as _Doc

        d = _Doc()

        v16 = finch_db.VectorSchema("image", finch_db.DataType.VECTOR_FP16, 0)
        d.set_any("image", v16._get_object(), [1.0, 2.0, 3.0])
        vec = d.get_any("image", finch_db.DataType.VECTOR_FP16)
        self.assertEqual(len(vec), 3)
        for i in range(3):
            self.assertTrue(math.isclose(vec[i], [1.0, 2.0, 3.0][i], rel_tol=1e-1))

        v32 = finch_db.VectorSchema("image2", finch_db.DataType.VECTOR_FP32, 0)
        d.set_any("image2", v32._get_object(), [1.1, 2.2, 3.3])
        vec2 = d.get_any("image2", finch_db.DataType.VECTOR_FP32)
        for i in range(3):
            self.assertTrue(math.isclose(vec2[i], [1.1, 2.2, 3.3][i], rel_tol=1e-6))

        vint8 = finch_db.VectorSchema("text", finch_db.DataType.VECTOR_INT8, 0)
        d.set_any("text", vint8._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("text", finch_db.DataType.VECTOR_INT8), [1, 2, 3])

        sparse = {1: 1.111111, 2: 2.222222, 3: 3.333333}
        vs = finch_db.VectorSchema("key", finch_db.DataType.SPARSE_VECTOR_FP32, 0)
        d.set_any("key", vs._get_object(), sparse)
        got = d.get_any("key", finch_db.DataType.SPARSE_VECTOR_FP32)
        self.assertIsInstance(got, dict)
        for k, v in sparse.items():
            self.assertTrue(math.isclose(got[k], v, rel_tol=1e-6))

    def test_convert_to_core_and_py_doc(self):
        from finch_db.model.convert import convert_to_core_doc, convert_to_py_doc
        from finch_db._finch import PyDoc as _Doc

        schema = finch_db.CollectionSchema(
            name="test_collection",
            fields=finch_db.FieldSchema("name", finch_db.DataType.STRING),
        )
        core = convert_to_core_doc(finch_db.Doc(id="1"), schema=schema)
        self.assertEqual(core.pk(), "1")

        schema2 = finch_db.CollectionSchema(
            name="test_collection",
            fields=[
                finch_db.FieldSchema("id", finch_db.DataType.UINT64),
                finch_db.FieldSchema("salary", finch_db.DataType.UINT32),
                finch_db.FieldSchema("age", finch_db.DataType.INT32),
                finch_db.FieldSchema("create_at", finch_db.DataType.INT64),
                finch_db.FieldSchema("author", finch_db.DataType.STRING),
                finch_db.FieldSchema("weight", finch_db.DataType.FLOAT),
                finch_db.FieldSchema("bmi", finch_db.DataType.DOUBLE),
                finch_db.FieldSchema("is_male", finch_db.DataType.BOOL),
            ],
        )
        doc = finch_db.Doc(
            id="1",
            fields={
                "id": 1,
                "salary": 1000,
                "age": 18,
                "create_at": 1640995200,
                "author": "Tom",
                "weight": 80.0,
                "bmi": 0.4,
                "is_male": True,
            },
        )
        core2 = convert_to_core_doc(doc, schema=schema2)
        self.assertEqual(core2.get_any("author", finch_db.DataType.STRING), "Tom")

        # Field not in schema => ValueError.
        bad_schema = finch_db.CollectionSchema(
            name="test_collection",
            fields=[finch_db.FieldSchema("id", finch_db.DataType.UINT64)],
        )
        with self.assertRaises(ValueError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"name": "Tom"}), schema=bad_schema)

        schema3 = finch_db.CollectionSchema(
            name="test_collection",
            fields=[
                finch_db.FieldSchema("ids_u64", finch_db.DataType.ARRAY_UINT64),
                finch_db.FieldSchema("ids_u32", finch_db.DataType.ARRAY_UINT32),
            ],
        )
        core3 = convert_to_core_doc(
            finch_db.Doc(id="1", fields={"ids_u64": [1, 2, 3], "ids_u32": [4, 5]}),
            schema=schema3,
        )
        self.assertEqual(core3.get_any("ids_u64", finch_db.DataType.ARRAY_UINT64), [1, 2, 3])
        self.assertEqual(core3.get_any("ids_u32", finch_db.DataType.ARRAY_UINT32), [4, 5])

        core = _Doc()
        core.set_pk("1")
        core.set_score(1.0)
        core.set_any("author", schema2.field("author")._get_object(), "Tom")
        py = convert_to_py_doc(core, schema2)
        self.assertEqual(py.id, "1")
        self.assertEqual(py.score, 1.0)
        self.assertEqual(py.field("author"), "Tom")
