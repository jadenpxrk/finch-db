from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


def _workflow_schema() -> finch_db.CollectionSchema:
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
        ],
        vectors=[
            finch_db.VectorSchema(
                "dense",
                finch_db.DataType.VECTOR_FP32,
                16,
                index_param=finch_db.HnswIndexParam(),
            ),
            finch_db.VectorSchema(
                "sparse",
                finch_db.DataType.SPARSE_VECTOR_FP32,
                0,
                index_param=finch_db.HnswIndexParam(),
            ),
        ],
    )


def _doc(np, doc_id: int, name: str, weight: float, sparse: dict[int, float]) -> finch_db.Doc:
    return finch_db.Doc(
        id=str(doc_id),
        fields={"id": doc_id, "name": name, "weight": weight},
        vectors={"dense": np.random.random(16).tolist(), "sparse": sparse},
    )


class CollectionWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_complicated_workflow(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_workflow_")
        path = f"{root}/collection"
        try:
            schema = _workflow_schema()

            opt = finch_db.CollectionOption(read_only=False, enable_mmap=True)
            col = finch_db.create_and_open(path=path, schema=schema, option=opt)

            self.assertEqual(col.path, path)
            self.assertEqual(col.schema.name, schema.name)
            self.assertEqual(col.stats.doc_count, 0)

            # Step 2: Create index (should rebuild/replace existing index params).
            col.create_index(field_name="name", index_param=finch_db.InvertIndexParam(), option=finch_db.IndexOption())
            _ = col.stats

            # Step 3: Insert doc1
            self.assertTrue(col.insert(_doc(np, 1, "test1", 80.5, {1: 1.0, 2: 2.0})).ok())
            self.assertEqual(col.stats.doc_count, 1)

            # Step 4: Upsert existing doc1
            self.assertTrue(col.upsert(_doc(np, 1, "test1_updated", 85.0, {1: 1.5, 2: 2.5})).ok())
            self.assertEqual(col.stats.doc_count, 1)

            # Step 5: Insert + update doc2
            self.assertTrue(col.insert(_doc(np, 2, "test2", 90.0, {1: 3.0, 2: 4.0})).ok())
            self.assertEqual(col.stats.doc_count, 2)

            self.assertTrue(col.update(_doc(np, 2, "test2_updated", 95.0, {1: 3.5, 2: 4.5})).ok())
            self.assertEqual(col.stats.doc_count, 2)

            # Step 6: Fetch
            fetched = col.fetch(["1", "2"])
            self.assertEqual(len(fetched), 2)
            self.assertEqual(fetched["1"].field("name"), "test1_updated")
            self.assertEqual(fetched["2"].field("name"), "test2_updated")

            # Step 7: Query
            q = col.query(filter="id >= 1", topk=10)
            self.assertEqual(len(q), 2)

            # Step 8: Drop index
            col.drop_index(field_name="name")

            # Step 9: Insert + update doc3
            self.assertTrue(col.insert(_doc(np, 3, "test3", 100.0, {1: 5.0, 2: 6.0})).ok())
            self.assertEqual(col.stats.doc_count, 3)

            self.assertTrue(col.update(_doc(np, 3, "test3_updated", 105.0, {1: 5.5, 2: 6.5})).ok())
            self.assertEqual(col.stats.doc_count, 3)

            # Step 11: Upsert doc4 (new)
            self.assertTrue(col.upsert(_doc(np, 4, "test4", 110.0, {1: 7.0, 2: 8.0})).ok())
            self.assertEqual(col.stats.doc_count, 4)

            # Step 12: Fetch doc3+doc4
            fetched = col.fetch(["3", "4"])
            self.assertEqual(len(fetched), 2)
            self.assertEqual(fetched["3"].field("name"), "test3_updated")
            self.assertEqual(fetched["4"].field("name"), "test4")

            # Step 13: Query doc3+doc4
            q = col.query(filter="id >= 3", topk=10)
            self.assertEqual(len(q), 2)

            # Step 14: Flush then re-fetch
            col.flush()
            fetched = col.fetch(["1", "2", "3", "4"])
            self.assertEqual(len(fetched), 4)

            # Step 15: Destroy
            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

