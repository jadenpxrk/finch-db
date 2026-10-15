from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


class CollectionDdlOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def _make_collection(self, path: str) -> finch_db.Collection:
        schema = finch_db.CollectionSchema(
            name="ddl_ops",
            fields=[
                finch_db.FieldSchema(
                    "id",
                    finch_db.DataType.INT64,
                    nullable=False,
                    index_param=finch_db.InvertIndexParam(enable_range_optimization=True),
                ),
                finch_db.FieldSchema("weight", finch_db.DataType.FLOAT, nullable=False),
                finch_db.FieldSchema("name", finch_db.DataType.STRING, nullable=False),
            ],
            vectors=[
                finch_db.VectorSchema("dense", finch_db.DataType.VECTOR_FP32, 8, index_param=finch_db.FlatIndexParam()),
                finch_db.VectorSchema("sparse", finch_db.DataType.SPARSE_VECTOR_FP32, 0, index_param=finch_db.FlatIndexParam()),
            ],
        )
        return finch_db.create_and_open(path, schema, finch_db.CollectionOption(read_only=False, enable_mmap=True))

    def test_create_drop_invert_index_preserves_filter_results(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_ddl_ops_")
        path = f"{root}/c"
        try:
            col = self._make_collection(path)
            docs = [
                finch_db.Doc(
                    id=str(i),
                    fields={"id": i, "weight": float(i), "name": f"n{i}"},
                    vectors={"dense": [float(i)] * 8, "sparse": {1: float(i + 1)}},
                )
                for i in range(10)
            ]
            res = col.insert(docs)
            self.assertTrue(all(s.is_ok() for s in res))

            before = [d.id for d in col.query(filter="weight >= 5.0", topk=50, output_fields=["weight"])]

            col.create_index("weight", finch_db.InvertIndexParam(enable_range_optimization=True), finch_db.IndexOption())
            after_create = [d.id for d in col.query(filter="weight >= 5.0", topk=50, output_fields=["weight"])]
            self.assertEqual(set(before), set(after_create))

            # A duplicate create_index rebuilds and replaces the index.
            col.create_index("weight", finch_db.InvertIndexParam(enable_range_optimization=False), finch_db.IndexOption())
            after_dup = [d.id for d in col.query(filter="weight >= 5.0", topk=50, output_fields=["weight"])]
            self.assertEqual(set(before), set(after_dup))

            col.drop_index("weight")
            after_drop = [d.id for d in col.query(filter="weight >= 5.0", topk=50, output_fields=["weight"])]
            self.assertEqual(set(before), set(after_drop))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_create_hnsw_index_rebuilds_vector_index(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_ddl_ops_hnsw_")
        path = f"{root}/c"
        try:
            col = self._make_collection(path)
            docs = []
            for i in range(30):
                v = [0.0] * 8
                v[i % 8] = 1.0
                docs.append(
                    finch_db.Doc(
                        id=str(i),
                        fields={"id": i, "weight": 1.0, "name": "x"},
                        vectors={"dense": v, "sparse": {1: 1.0}},
                    )
                )
            res = col.insert(docs)
            self.assertTrue(all(s.is_ok() for s in res))
            col.flush()

            q = finch_db.VectorQuery("dense", vector=[1.0] + [0.0] * 7)
            before = col.query(vectors=q, topk=5, output_fields=["id"])
            self.assertTrue(len(before) > 0)

            col.create_index("dense", finch_db.HnswIndexParam(metric=finch_db.MetricType.L2), finch_db.IndexOption(concurrency=0))
            col.optimize(finch_db.OptimizeOption(concurrency=0))

            after = col.query(vectors=q, topk=5, output_fields=["id"])
            self.assertTrue(len(after) > 0)
            # Top-1 should remain in the same cluster.
            self.assertEqual(int(after[0].id) % 8, 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_delete_by_filter_deletes_docs(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_ddl_ops_del_filter_")
        path = f"{root}/c"
        try:
            col = self._make_collection(path)
            docs = [
                finch_db.Doc(
                    id=str(i),
                    fields={"id": i, "weight": float(i), "name": f"n{i}"},
                    vectors={"dense": [0.0] * 8, "sparse": {1: 1.0}},
                )
                for i in range(10)
            ]
            res = col.insert(docs)
            self.assertTrue(all(s.is_ok() for s in res))
            self.assertEqual(col.stats.doc_count, 10)

            col.delete_by_filter("id >= 5")
            self.assertEqual(col.stats.doc_count, 5)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
