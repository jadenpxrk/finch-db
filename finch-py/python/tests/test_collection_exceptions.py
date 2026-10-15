from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


class CollectionExceptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def _make_collection(self, path: str) -> finch_db.Collection:
        schema = finch_db.CollectionSchema(
            name="exceptions",
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
        return finch_db.create_and_open(path, schema, finch_db.CollectionOption(read_only=False, enable_mmap=True))

    def test_missing_required_parameters_raise(self) -> None:
        with self.assertRaises(TypeError):
            finch_db.create_and_open(schema=finch_db.CollectionSchema("x"))  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            finch_db.create_and_open(path="/tmp/x")  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            finch_db.open()  # type: ignore[call-arg]

        root = tempfile.mkdtemp(prefix="finch_test_exception_")
        path = f"{root}/collection"
        try:
            col = self._make_collection(path)
            with self.assertRaises(TypeError):
                col.insert()  # type: ignore[call-arg]
            with self.assertRaises(TypeError):
                col.update()  # type: ignore[call-arg]
            with self.assertRaises(TypeError):
                col.upsert()  # type: ignore[call-arg]
            with self.assertRaises(TypeError):
                col.delete()  # type: ignore[call-arg]
            with self.assertRaises(TypeError):
                col.fetch()  # type: ignore[call-arg]
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_empty_collection_operations_do_not_crash(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_empty_ops_")
        path = f"{root}/collection"
        try:
            col = self._make_collection(path)

            got = col.fetch(["1"])
            self.assertIsInstance(got, dict)

            out = col.query()
            self.assertEqual(out, [])

            d = finch_db.Doc(
                id="1",
                fields={"id": 1, "name": "test"},
                vectors={"dense": np.random.random(8).tolist()},
            )
            st = col.update(d)
            self.assertTrue(hasattr(st, "ok"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_query_missing_vector_field_name_raises(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_query_missing_field_")
        path = f"{root}/collection"
        try:
            col = self._make_collection(path)
            with self.assertRaises(Exception):
                col.query(vectors=[finch_db.VectorQuery()])  # type: ignore[call-arg]
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

