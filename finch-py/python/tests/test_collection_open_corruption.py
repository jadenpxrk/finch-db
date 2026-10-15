from __future__ import annotations

import gc
import os
import random
import shutil
import tempfile
import unittest

import finch_db


def _schema() -> finch_db.CollectionSchema:
    return finch_db.CollectionSchema(
        name="corruption",
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


class CollectionOpenCorruptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_open_with_corrupted_files_raises(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_open_corrupt_")
        path = f"{root}/collection"
        opt = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        try:
            col = finch_db.create_and_open(path=path, schema=_schema(), option=opt)
            d = finch_db.Doc(
                id="1",
                fields={"id": 1, "name": "test"},
                vectors={"dense": np.random.random(8).tolist()},
            )
            self.assertTrue(col.insert(d).ok())
            col.flush()
            col.close()
            del col
            gc.collect()

            # Delete ~half of files under collection directory.
            files: list[str] = []
            for r, _dirs, fs in os.walk(path):
                for f in fs:
                    files.append(os.path.join(r, f))
            random.Random(0).shuffle(files)
            for fp in files[: max(1, len(files) // 2)]:
                try:
                    os.remove(fp)
                except Exception:
                    pass

            with self.assertRaises(Exception):
                finch_db.open(path=path, option=opt)

            # Case 2: empty dir should fail to open.
            empty_path = f"{root}/empty"
            os.makedirs(empty_path, exist_ok=True)
            with self.assertRaises(Exception):
                finch_db.open(path=empty_path, option=opt)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

