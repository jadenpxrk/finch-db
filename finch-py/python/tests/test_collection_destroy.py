from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


class CollectionDestroyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_destroy_makes_collection_unusable_and_removes_path(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_destroy_")
        path = f"{root}/collection"
        opt = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        try:
            schema = finch_db.CollectionSchema(
                name="destroy",
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
                ],
                vectors=[
                    finch_db.VectorSchema(
                        "dense",
                        finch_db.DataType.VECTOR_FP32,
                        8,
                        index_param=finch_db.HnswIndexParam(),
                    ),
                ],
            )
            col = finch_db.create_and_open(path=path, schema=schema, option=opt)
            st = col.insert(
                finch_db.Doc(
                    id="1",
                    fields={"id": 1, "name": "x"},
                    vectors={"dense": np.random.random(8).tolist()},
                )
            )
            self.assertTrue(st.ok())
            self.assertEqual(col.stats.doc_count, 1)

            col.destroy()

            with self.assertRaises(Exception):
                _ = col.stats
            with self.assertRaises(Exception):
                _ = col.schema
            with self.assertRaises(Exception):
                finch_db.open(path=path, option=opt)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
