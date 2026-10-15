from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


def _schema_hnsw_all() -> finch_db.CollectionSchema:
    # Minimal schema to exercise both dense + sparse query-param plumbing.
    return finch_db.CollectionSchema(
        name="query_param",
        fields=[
            finch_db.FieldSchema(
                "id",
                finch_db.DataType.INT32,
                nullable=False,
                index_param=finch_db.InvertIndexParam(enable_range_optimization=True),
            )
        ],
        vectors=[
            finch_db.VectorSchema("v_fp32", finch_db.DataType.VECTOR_FP32, 8, index_param=finch_db.HnswIndexParam()),
            finch_db.VectorSchema("v_fp16", finch_db.DataType.VECTOR_FP16, 8, index_param=finch_db.HnswIndexParam()),
            finch_db.VectorSchema("v_i8", finch_db.DataType.VECTOR_INT8, 8, index_param=finch_db.HnswIndexParam()),
            finch_db.VectorSchema("v_s32", finch_db.DataType.SPARSE_VECTOR_FP32, 0, index_param=finch_db.HnswIndexParam()),
            finch_db.VectorSchema("v_s16", finch_db.DataType.SPARSE_VECTOR_FP16, 0, index_param=finch_db.HnswIndexParam()),
        ],
    )


def _schema_ivf_fp32() -> finch_db.CollectionSchema:
    return finch_db.CollectionSchema(
        name="query_param_ivf",
        fields=[
            finch_db.FieldSchema(
                "id",
                finch_db.DataType.INT32,
                nullable=False,
                index_param=finch_db.InvertIndexParam(enable_range_optimization=True),
            )
        ],
        vectors=[
            finch_db.VectorSchema("v", finch_db.DataType.VECTOR_FP32, 8, index_param=finch_db.IVFIndexParam()),
        ],
    )


def _doc(i: int) -> finch_db.Doc:
    return finch_db.Doc(
        id=str(i),
        fields={"id": i},
        vectors={
            "v_fp32": [float(i + 1)] * 8,
            "v_fp16": [float(i + 1)] * 8,
            "v_i8": [int(i + 1)] * 8,
            "v_s32": {i: i + 0.1},
            "v_s16": {i: i + 0.1},
        },
    )


class CollectionQueryParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_query_vector_with_hnsw_query_param_valid(self) -> None:
        # Valid HnswQueryParam values.
        root = tempfile.mkdtemp(prefix="finch_test_qparam_hnsw_")
        path = f"{root}/collection"
        try:
            col = finch_db.create_and_open(path, _schema_hnsw_all(), finch_db.CollectionOption(read_only=False, enable_mmap=True))
            res = col.insert([_doc(i) for i in range(10)])
            self.assertTrue(all(s.ok() for s in res))

            for ef in [0, 100, 1024, 2048]:
                with self.subTest(ef=ef):
                    q = finch_db.VectorQuery(field_name="v_fp32", vector=[1.0] * 8, param=finch_db.HnswQueryParam(ef=ef))
                    out = col.query(q, filter="id>=3 and id<=7", topk=10)
                    self.assertGreater(len(out), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_hnsw_query_param_invalid_constructor_types(self) -> None:
        # Invalid HnswQueryParam values.
        for ef in [None, "invalid", 10.5]:
            with self.subTest(ef=ef):
                with self.assertRaises(TypeError) as ctx:
                    finch_db.HnswQueryParam(ef=ef)  # type: ignore[arg-type]
                self.assertIn("incompatible constructor arguments", str(ctx.exception))

    def test_query_vector_with_ivf_query_param_valid(self) -> None:
        # Valid IVFQueryParam values.
        root = tempfile.mkdtemp(prefix="finch_test_qparam_ivf_")
        path = f"{root}/collection"
        try:
            col = finch_db.create_and_open(path, _schema_ivf_fp32(), finch_db.CollectionOption(read_only=False, enable_mmap=True))
            docs = [
                finch_db.Doc(id=str(i), fields={"id": i}, vectors={"v": [float(i + 1)] * 8})
                for i in range(10)
            ]
            res = col.insert(docs)
            self.assertTrue(all(s.ok() for s in res))
            col.optimize()

            for nprobe in [1, 10, 100, 2048]:
                with self.subTest(nprobe=nprobe):
                    q = finch_db.VectorQuery(field_name="v", vector=[1.0] * 8, param=finch_db.IVFQueryParam(nprobe=nprobe))
                    out = col.query(q, topk=10)
                    self.assertGreater(len(out), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_ivf_query_param_invalid_constructor_types(self) -> None:
        # Invalid IVFQueryParam values.
        for nprobe in [None, 10.5]:
            with self.subTest(nprobe=nprobe):
                with self.assertRaises(TypeError) as ctx:
                    finch_db.IVFQueryParam(nprobe=nprobe)  # type: ignore[arg-type]
                self.assertIn("incompatible constructor arguments", str(ctx.exception))

    def test_query_vector_with_param_invalid_type(self) -> None:
        # query.param must be a QueryParam.
        root = tempfile.mkdtemp(prefix="finch_test_qparam_wrongtype_")
        path = f"{root}/collection"
        try:
            col = finch_db.create_and_open(path, _schema_hnsw_all(), finch_db.CollectionOption(read_only=False, enable_mmap=True))
            res = col.insert([_doc(i) for i in range(10)])
            self.assertTrue(all(s.ok() for s in res))

            with self.assertRaises(TypeError) as ctx:
                col.query(
                    finch_db.VectorQuery(
                        field_name="v_fp32",
                        vector=[1.0] * 8,
                        param=finch_db.HnswIndexParam(),  # wrong type
                    ),
                    filter="id>=3 and id<=7",
                    topk=10,
                )
            self.assertIn("incompatible function arguments", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

