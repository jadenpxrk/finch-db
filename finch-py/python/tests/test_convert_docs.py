from __future__ import annotations

import math
import unittest

import finch_db
from finch_db.model.convert import convert_to_core_doc, convert_to_py_doc


class ConvertDocsTest(unittest.TestCase):
    def test_convert_to_core_doc_default(self):
        doc = finch_db.Doc(id="1")
        schema = finch_db.CollectionSchema(
            name="test_collection",
            fields=finch_db.FieldSchema("name", finch_db.DataType.STRING),
        )
        core_doc = convert_to_core_doc(doc, schema=schema)
        self.assertIsNotNone(core_doc)
        self.assertEqual(core_doc.pk(), "1")

    def test_convert_to_core_doc_field_not_in_schema_message(self):
        doc = finch_db.Doc(id="1", fields={"name": "Tom"})
        schema = finch_db.CollectionSchema(
            name="test_collection",
            fields=[finch_db.FieldSchema("id", finch_db.DataType.UINT64)],
        )
        with self.assertRaises(ValueError) as ctx:
            convert_to_core_doc(doc, schema=schema)
        self.assertIn("schema validate failed:", str(ctx.exception))

    def test_convert_to_core_doc_scalars_and_arrays(self):
        schema = finch_db.CollectionSchema(
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
                finch_db.FieldSchema("tags", finch_db.DataType.ARRAY_STRING),
                finch_db.FieldSchema("ids", finch_db.DataType.ARRAY_UINT64),
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
                "tags": ["tag1", "tag2"],
                "ids": [1, 2, 3],
            },
        )
        core = convert_to_core_doc(doc, schema=schema)
        self.assertEqual(core.get_any("id", finch_db.DataType.UINT64), 1)
        self.assertEqual(core.get_any("tags", finch_db.DataType.ARRAY_STRING), ["tag1", "tag2"])
        got_weight = core.get_any("weight", finch_db.DataType.FLOAT)
        self.assertTrue(math.isclose(got_weight, 80.0, rel_tol=1e-6))

    def test_convert_to_py_doc_uses_get_all(self):
        schema = finch_db.CollectionSchema(
            name="c",
            fields=[finch_db.FieldSchema("x", finch_db.DataType.INT32)],
            vectors=[finch_db.VectorSchema("v", finch_db.DataType.VECTOR_FP32, 2)],
        )
        core = convert_to_core_doc(
            finch_db.Doc(id="1", fields={"x": 7}, vectors={"v": [0.1, 0.2]}),
            schema=schema,
        )
        py = convert_to_py_doc(core, schema)
        self.assertEqual(py.id, "1")
        self.assertEqual(py.field("x"), 7)
        self.assertEqual(len(py.vector("v")), 2)
