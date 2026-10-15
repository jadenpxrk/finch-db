from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


def _make_schema(vector_name: str, vector_type: finch_db.DataType) -> finch_db.CollectionSchema:
    dim = 8 if vector_type not in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32) else 0
    return finch_db.CollectionSchema(
        name="vector_index_ddl",
        fields=[
            finch_db.FieldSchema(
                "id",
                finch_db.DataType.INT64,
                nullable=False,
                index_param=finch_db.InvertIndexParam(enable_range_optimization=True),
            )
        ],
        vectors=[
            finch_db.VectorSchema(
                vector_name,
                vector_type,
                dim,
                index_param=None,
            )
        ],
    )


def _doc(i: int, vector_name: str, vector_type: finch_db.DataType) -> finch_db.Doc:
    if vector_type in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32):
        vec = {1: float(i + 1), 2: float(i + 2), 3: float(i + 3)}
    elif vector_type == finch_db.DataType.VECTOR_INT8:
        vec = [int(i)] * 8
    else:
        vec = [float(i)] * 8
    return finch_db.Doc(
        id=str(i),
        fields={"id": i},
        vectors={vector_name: vec},
    )


class CollectionVectorIndexDdlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_vector_index_operations(self) -> None:
        # Every supported pair of vector data type and index type.
        cases: list[tuple[finch_db.DataType, finch_db.IndexType]] = [
            (finch_db.DataType.VECTOR_FP16, finch_db.IndexType.FLAT),
            (finch_db.DataType.VECTOR_FP16, finch_db.IndexType.HNSW),
            (finch_db.DataType.VECTOR_FP16, finch_db.IndexType.IVF),
            (finch_db.DataType.VECTOR_FP32, finch_db.IndexType.FLAT),
            (finch_db.DataType.VECTOR_FP32, finch_db.IndexType.HNSW),
            (finch_db.DataType.VECTOR_FP32, finch_db.IndexType.IVF),
            (finch_db.DataType.VECTOR_INT8, finch_db.IndexType.FLAT),
            (finch_db.DataType.VECTOR_INT8, finch_db.IndexType.HNSW),
            (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.IndexType.FLAT),
            (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.IndexType.HNSW),
            (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.IndexType.FLAT),
            (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.IndexType.HNSW),
        ]

        for vector_type, index_type in cases:
            with self.subTest(vector_type=vector_type.name, index_type=index_type.name):
                root = tempfile.mkdtemp(prefix="finch_test_vec_index_")
                path = f"{root}/collection"
                try:
                    vector_name = "v"
                    schema = _make_schema(vector_name, vector_type)
                    col = finch_db.create_and_open(path, schema, finch_db.CollectionOption(read_only=False, enable_mmap=True))

                    res = col.insert([_doc(i, vector_name, vector_type) for i in range(5)])
                    self.assertTrue(all(s.is_ok() for s in res))
                    self.assertEqual(col.stats.doc_count, 5)

                    if index_type == finch_db.IndexType.FLAT:
                        index_param = finch_db.FlatIndexParam()
                    elif index_type == finch_db.IndexType.HNSW:
                        index_param = finch_db.HnswIndexParam()
                    elif index_type == finch_db.IndexType.IVF:
                        index_param = finch_db.IVFIndexParam()
                    else:  # pragma: no cover
                        raise AssertionError(f"unsupported index type: {index_type}")

                    col.create_index(vector_name, index_param, finch_db.IndexOption())

                    res2 = col.insert([_doc(i, vector_name, vector_type) for i in range(5, 8)])
                    self.assertTrue(all(s.is_ok() for s in res2))
                    self.assertEqual(col.stats.doc_count, 8)
                    fetched = col.fetch([str(i) for i in range(5, 8)])
                    self.assertEqual(len(fetched), 3)

                    col.drop_index(vector_name)

                    res3 = col.insert([_doc(i, vector_name, vector_type) for i in range(8, 10)])
                    self.assertTrue(all(s.is_ok() for s in res3))
                    self.assertEqual(col.stats.doc_count, 10)
                    fetched2 = col.fetch([str(i) for i in range(8, 10)])
                    self.assertEqual(len(fetched2), 2)
                finally:
                    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

