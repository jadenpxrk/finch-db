from __future__ import annotations

import tempfile
import unittest

import finch_db


class CollectionInvalidVectorIndexParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_invalid_vector_index_params_in_schema(self) -> None:
        # Pairs of vector data type and index parameters that schema validation rejects.
        cases: list[tuple[finch_db.DataType, object]] = [
            (finch_db.DataType.VECTOR_FP32, finch_db.InvertIndexParam()),
            (finch_db.DataType.VECTOR_FP16, finch_db.InvertIndexParam()),
            (finch_db.DataType.VECTOR_INT8, finch_db.InvertIndexParam()),
            (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.L2)),
            (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.COSINE)),
            (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.IVFIndexParam()),
            (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.InvertIndexParam()),
            (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.L2)),
            (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.COSINE)),
            (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.IVFIndexParam()),
            (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.InvertIndexParam()),
        ]

        for data_type, index_param in cases:
            with self.subTest(data_type=data_type.name, index_param=type(index_param).__name__):
                dim = 0 if data_type in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32) else 8
                schema = finch_db.CollectionSchema(
                    name="invalid_vec_param",
                    fields=[finch_db.FieldSchema("id", finch_db.DataType.INT64, nullable=False)],
                    vectors=[finch_db.VectorSchema("v", data_type=data_type, dimension=dim, index_param=index_param)],
                )
                with tempfile.TemporaryDirectory(prefix="finch_test_invalid_vec_") as td:
                    path = f"{td}/collection"
                    with self.assertRaises(ValueError):
                        finch_db.create_and_open(path, schema, finch_db.CollectionOption(read_only=False, enable_mmap=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)

