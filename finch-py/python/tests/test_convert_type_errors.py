from __future__ import annotations

import unittest

import finch_db
from finch_db.model.convert import convert_to_core_doc


class ConvertTypeErrorsTest(unittest.TestCase):
    def test_scalar_fields_type_errors(self):
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
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"id": "1"}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"salary": "1000"}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"age": "18"}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"create_at": "2021-01-01"}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"author": 1}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"weight": "80.5"}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"bmi": "25.0"}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"is_male": "true"}), schema=schema)

    def test_array_fields_type_errors(self):
        schema = finch_db.CollectionSchema(
            name="test_collection",
            fields=[
                finch_db.FieldSchema("tags", finch_db.DataType.ARRAY_STRING),
                finch_db.FieldSchema("ids", finch_db.DataType.ARRAY_UINT64),
                finch_db.FieldSchema("marks", finch_db.DataType.ARRAY_UINT32),
                finch_db.FieldSchema("x", finch_db.DataType.ARRAY_INT32),
                finch_db.FieldSchema("y", finch_db.DataType.ARRAY_INT64),
                finch_db.FieldSchema("scores", finch_db.DataType.ARRAY_FLOAT),
                finch_db.FieldSchema("ratios", finch_db.DataType.ARRAY_DOUBLE),
                finch_db.FieldSchema("results", finch_db.DataType.ARRAY_BOOL),
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"tags": [1, 2, 3]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"ids": ["1", "2", "3"]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"marks": [1.1, 2.2, 3.3]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"x": [1.1, 2.2, 3.3]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"y": [1.1, 2.2, 3.3]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"scores": ["1", "2", "3"]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"ratios": ["1", "2", "3"]}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", fields={"results": ["1", "2", "3"]}), schema=schema)

    def test_vector_fields_type_errors(self):
        schema = finch_db.CollectionSchema(
            name="test_collection",
            vectors=[
                finch_db.VectorSchema(name="embedding", data_type=finch_db.DataType.VECTOR_FP16, dimension=4),
                finch_db.VectorSchema(name="image", data_type=finch_db.DataType.VECTOR_FP32, dimension=8),
                finch_db.VectorSchema(name="text", data_type=finch_db.DataType.VECTOR_INT8, dimension=32),
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", vectors={"image": ["1.1"] * 4}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", vectors={"text": ["1"] * 4}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", vectors={"embedding": ["1"] * 4}), schema=schema)

    def test_sparse_vector_type_errors(self):
        schema = finch_db.CollectionSchema(
            name="test_collection",
            vectors=[
                finch_db.VectorSchema(name="author", data_type=finch_db.DataType.SPARSE_VECTOR_FP32),
                finch_db.VectorSchema(name="content", data_type=finch_db.DataType.SPARSE_VECTOR_FP16),
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", vectors={"author": {"1": 1.1}}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", vectors={"content": {"1": 1.1}}), schema=schema)
        with self.assertRaises(TypeError):
            convert_to_core_doc(finch_db.Doc(id="1", vectors={"author": {1: "1"}}), schema=schema)

