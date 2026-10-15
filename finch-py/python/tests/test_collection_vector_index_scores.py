from __future__ import annotations

import math
import shutil
import tempfile
import unittest

import numpy as np

import finch_db


def _is_float_equal(actual: float | None, expected: float | None, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> bool:
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def _generate_constant_vector(i: int, dimension: int, dtype: str) -> list[float] | list[int]:
    if dtype == "int8":
        vec = [i % 128] * dimension
        vec[i % dimension] = (i + 1) % 128
        return vec

    vec = [i / 256.0] * dimension
    vec[i % dimension] = (i + 1) / 256.0
    return vec


def _distance_dense(
    vec1: list[float] | list[int],
    vec2: list[float] | list[int],
    metric: finch_db.MetricType,
    data_type: finch_db.DataType,
    quantize_type: finch_db.QuantizeType,
) -> float:
    wants_fp16 = data_type == finch_db.DataType.VECTOR_FP16 or quantize_type == finch_db.QuantizeType.FP16

    if metric == finch_db.MetricType.COSINE:
        if wants_fp16:
            v1 = [np.float16(x) for x in vec1]  # type: ignore[list-item]
            v2 = [np.float16(x) for x in vec2]  # type: ignore[list-item]
        else:
            v1 = [float(x) for x in vec1]
            v2 = [float(x) for x in vec2]

        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return float(1 - dot_product / (magnitude1 * magnitude2))

    if metric == finch_db.MetricType.L2:
        if wants_fp16:
            return float(sum((np.float16(a) - np.float16(b)) ** 2 for a, b in zip(vec1, vec2)))  # type: ignore[arg-type]
        return float(sum((float(a) - float(b)) ** 2 for a, b in zip(vec1, vec2)))

    if metric == finch_db.MetricType.IP:
        if wants_fp16:
            return float(sum(np.float16(a) * np.float16(b) for a, b in zip(vec1, vec2)))  # type: ignore[arg-type]
        return float(sum(float(a) * float(b) for a, b in zip(vec1, vec2)))

    if metric == finch_db.MetricType.MIPS_L2:
        # MIPS_L2 with localized spherical injection (e2 == 0.0):
        #   dist = 2 - 2 * ip(a,b) / max(||a||^2, ||b||^2)
        if wants_fp16:
            v1 = [np.float16(x) for x in vec1]  # type: ignore[list-item]
            v2 = [np.float16(x) for x in vec2]  # type: ignore[list-item]
            ip = float(sum(a * b for a, b in zip(v1, v2)))
            u2 = float(sum(a * a for a in v1))
            v2s = float(sum(b * b for b in v2))
        else:
            ip = float(sum(float(a) * float(b) for a, b in zip(vec1, vec2)))
            u2 = float(sum(float(a) * float(a) for a in vec1))
            v2s = float(sum(float(b) * float(b) for b in vec2))
        denom = max(u2, v2s)
        return float(2 - 2 * ip / denom)

    raise ValueError(f"unsupported metric: {metric}")


def _distance_sparse(
    vec1: dict[int, float],
    vec2: dict[int, float],
    data_type: finch_db.DataType,
    quantize_type: finch_db.QuantizeType,
) -> float:
    wants_fp16 = data_type == finch_db.DataType.SPARSE_VECTOR_FP16 or quantize_type == finch_db.QuantizeType.FP16
    dot_product: float | np.float16 = 0.0
    for dim in set(vec1.keys()) & set(vec2.keys()):
        a = vec1[dim]
        b = vec2[dim]
        if wants_fp16:
            a = np.float16(a)  # type: ignore[assignment]
            b = np.float16(b)  # type: ignore[assignment]
        dot_product += a * b  # type: ignore[operator]
    return float(dot_product)


def _distance(
    vec1,
    vec2,
    metric: finch_db.MetricType,
    data_type: finch_db.DataType,
    quantize_type: finch_db.QuantizeType,
) -> float:
    if data_type in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32):
        if metric != finch_db.MetricType.IP:
            raise ValueError("unsupported metric type for sparse vectors")
        return _distance_sparse(vec1, vec2, data_type, quantize_type)
    return _distance_dense(vec1, vec2, metric, data_type, quantize_type)


def _make_schema(vector_name: str, data_type: finch_db.DataType, dim: int) -> finch_db.CollectionSchema:
    return finch_db.CollectionSchema(
        name="vector_index_params",
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
                data_type=data_type,
                dimension=dim,
                index_param=None,
            )
        ],
    )


def _doc(i: int, vector_name: str, data_type: finch_db.DataType, dim: int) -> finch_db.Doc:
    if data_type in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32):
        vec = {i: i + 0.1}
    elif data_type == finch_db.DataType.VECTOR_INT8:
        vec = _generate_constant_vector(i + 1, dim, "int8")
    else:
        vec = _generate_constant_vector(i + 1, dim, "float")
    return finch_db.Doc(id=str(i), fields={"id": i}, vectors={vector_name: vec})


def _index_param_cases() -> list[tuple[finch_db.DataType, object]]:
    # Every valid pair of vector data type and index parameters.
    return [
        # VECTOR_FP32
        (finch_db.DataType.VECTOR_FP32, finch_db.HnswIndexParam()),
        (finch_db.DataType.VECTOR_FP32, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.IP, m=16, ef_construction=100, quantize_type=finch_db.QuantizeType.INT8)),
        (finch_db.DataType.VECTOR_FP32, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.COSINE, m=24, ef_construction=150, quantize_type=finch_db.QuantizeType.INT4)),
        (finch_db.DataType.VECTOR_FP32, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.L2, m=32, ef_construction=200, quantize_type=finch_db.QuantizeType.FP16)),
        (finch_db.DataType.VECTOR_FP32, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.MIPS_L2, m=32, ef_construction=200, quantize_type=finch_db.QuantizeType.FP16)),
        (finch_db.DataType.VECTOR_FP32, finch_db.FlatIndexParam()),
        (finch_db.DataType.VECTOR_FP32, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.IP, quantize_type=finch_db.QuantizeType.INT4)),
        (finch_db.DataType.VECTOR_FP32, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.L2, quantize_type=finch_db.QuantizeType.INT8)),
        (finch_db.DataType.VECTOR_FP32, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.COSINE, quantize_type=finch_db.QuantizeType.FP16)),
        (finch_db.DataType.VECTOR_FP32, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.MIPS_L2, quantize_type=finch_db.QuantizeType.UNDEFINED)),
        (finch_db.DataType.VECTOR_FP32, finch_db.FlatIndexParam(metric_type=finch_db.MetricType.MIPS_L2, quantize_type=finch_db.QuantizeType.FP16)),
        (finch_db.DataType.VECTOR_FP32, finch_db.IVFIndexParam()),
        (finch_db.DataType.VECTOR_FP32, finch_db.IVFIndexParam(metric_type=finch_db.MetricType.IP, quantize_type=finch_db.QuantizeType.INT4, n_list=100, n_iters=10, use_soar=False)),
        (finch_db.DataType.VECTOR_FP32, finch_db.IVFIndexParam(metric_type=finch_db.MetricType.L2, quantize_type=finch_db.QuantizeType.INT8, n_list=200, n_iters=20, use_soar=True)),
        (finch_db.DataType.VECTOR_FP32, finch_db.IVFIndexParam(metric_type=finch_db.MetricType.COSINE, quantize_type=finch_db.QuantizeType.FP16, n_list=150, n_iters=15, use_soar=False)),
        (finch_db.DataType.VECTOR_FP32, finch_db.IVFIndexParam(metric_type=finch_db.MetricType.MIPS_L2, quantize_type=finch_db.QuantizeType.FP16, n_list=150, n_iters=15, use_soar=False)),
        # VECTOR_FP16
        (finch_db.DataType.VECTOR_FP16, finch_db.HnswIndexParam()),
        (finch_db.DataType.VECTOR_FP16, finch_db.FlatIndexParam()),
        # VECTOR_INT8
        (finch_db.DataType.VECTOR_INT8, finch_db.HnswIndexParam()),
        (finch_db.DataType.VECTOR_INT8, finch_db.FlatIndexParam()),
        # SPARSE_VECTOR_FP32
        (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.HnswIndexParam()),
        (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.FlatIndexParam()),
        (finch_db.DataType.SPARSE_VECTOR_FP32, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.IP, m=16, ef_construction=100, quantize_type=finch_db.QuantizeType.FP16)),
        # SPARSE_VECTOR_FP16
        (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.HnswIndexParam()),
        (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.FlatIndexParam()),
        (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.HnswIndexParam(metric_type=finch_db.MetricType.IP, m=16, ef_construction=100)),
    ]


class CollectionVectorIndexScoresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_vector_index_params_scores(self) -> None:
        for data_type, index_param in _index_param_cases():
            with self.subTest(data_type=data_type.name, index_param=type(index_param).__name__):
                self._check_index_param_scores(data_type, index_param)

    def _check_index_param_scores(self, data_type: finch_db.DataType, index_param) -> None:
        vector_name = "v"
        dim = 16
        root = tempfile.mkdtemp(prefix="finch_test_vec_params_")
        path = f"{root}/collection"
        try:
            schema_dim = 0 if data_type in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32) else dim
            schema = _make_schema(vector_name, data_type, schema_dim)

            col = finch_db.create_and_open(path, schema, finch_db.CollectionOption(read_only=False, enable_mmap=True))
            docs = {str(i): _doc(i, vector_name, data_type, dim) for i in range(5)}

            res = col.insert(list(docs.values()))
            self.assertEqual(len(res), len(docs))
            self.assertTrue(all(s.ok() for s in res))

            def check_result(label: str, metric_type: finch_db.MetricType, quantize_type: finch_db.QuantizeType) -> None:
                query_vector = [1] * dim
                if data_type in (finch_db.DataType.SPARSE_VECTOR_FP16, finch_db.DataType.SPARSE_VECTOR_FP32):
                    query_vector = {1: 1}  # type: ignore[assignment]

                out = col.query(
                    finch_db.VectorQuery(field_name=vector_name, vector=query_vector),
                    include_vector=False,
                    topk=len(docs),
                )
                self.assertEqual(len(out), len(docs), f"{label}: wrong result size")

                last_score: float | None = None
                for i, doc in enumerate(out):
                    expect_doc = docs[doc.id]
                    doc_vec = expect_doc.vector(vector_name)
                    expected_score = _distance(doc_vec, query_vector, metric_type, data_type, quantize_type)  # type: ignore[arg-type]

                    if quantize_type == finch_db.QuantizeType.UNDEFINED:
                        self.assertTrue(
                            _is_float_equal(doc.score, expected_score),
                            f"{label} top{i} pk{doc.id} score {doc.score} expected {expected_score}",
                        )

                    if last_score is not None:
                        if metric_type == finch_db.MetricType.IP:
                            self.assertGreaterEqual(
                                last_score,
                                doc.score,
                                f"{label}: score not sorted (IP): last {last_score}, current {doc.score}",
                            )
                        else:
                            self.assertLessEqual(
                                last_score,
                                doc.score,
                                f"{label}: score not sorted: last {last_score}, current {doc.score}",
                            )
                    last_score = doc.score

            # Default metric_type=IP, quantize_type=UNDEFINED
            check_result("pre_create_index", finch_db.MetricType.IP, finch_db.QuantizeType.UNDEFINED)

            col.create_index(vector_name, index_param, finch_db.IndexOption())
            check_result("post_create_index", index_param.metric_type, index_param.quantize_type)

            col.drop_index(vector_name)
            check_result("post_drop_index", finch_db.MetricType.IP, finch_db.QuantizeType.UNDEFINED)

            new_docs = {str(i): _doc(i, vector_name, data_type, dim) for i in range(5, 8)}
            new_res = col.insert(list(new_docs.values()))
            self.assertEqual(len(new_res), len(new_docs))
            self.assertTrue(all(s.ok() for s in new_res))
            docs |= new_docs

            col.create_index(vector_name, index_param, finch_db.IndexOption())
            check_result("post_create_index2", index_param.metric_type, index_param.quantize_type)

            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
