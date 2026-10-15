from __future__ import annotations

import shutil
import tempfile
import unittest

import finch_db


def _schema() -> finch_db.CollectionSchema:
    return finch_db.CollectionSchema(
        name="column_ddl",
        fields=[
            finch_db.FieldSchema("id", finch_db.DataType.INT64, nullable=False),
            finch_db.FieldSchema("name", finch_db.DataType.STRING, nullable=True),
            finch_db.FieldSchema("weight", finch_db.DataType.FLOAT, nullable=True),
        ],
        vectors=[
            finch_db.VectorSchema("dense", finch_db.DataType.VECTOR_FP32, 4),
        ],
    )


class CollectionColumnDdlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_drop_column_removes_field_from_schema_and_fetch(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_column_ddl_")
        path = f"{root}/collection"
        try:
            col = finch_db.create_and_open(path, _schema(), finch_db.CollectionOption(read_only=False, enable_mmap=True))

            col.add_column(
                field_schema=finch_db.FieldSchema("temp_field", finch_db.DataType.INT32, nullable=True),
                expression="100",
            )

            d1 = finch_db.Doc(
                id="1",
                fields={"id": 1, "name": "x", "weight": 1.0, "temp_field": 123},
                vectors={"dense": [0.1, 0.2, 0.3, 0.4]},
            )
            s = col.insert(d1)
            self.assertTrue(s.ok())
            col.flush()

            got = col.fetch("1")["1"]
            self.assertIn("temp_field", got.fields)

            col.drop_column("temp_field")

            got2 = col.fetch("1")["1"]
            self.assertNotIn("temp_field", got2.fields)

            # Inserting a dropped field should fail schema validation.
            with self.assertRaises(Exception):
                col.insert(
                    finch_db.Doc(
                        id="2",
                        fields={"id": 2, "name": "y", "weight": 2.0, "temp_field": 1},
                        vectors={"dense": [0.1, 0.2, 0.3, 0.4]},
                    )
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_drop_column_missing_raises(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_drop_missing_")
        path = f"{root}/collection"
        try:
            col = finch_db.create_and_open(path, _schema(), finch_db.CollectionOption(read_only=False, enable_mmap=True))
            with self.assertRaises(Exception) as ctx:
                col.drop_column("non_existing_column")
            self.assertIn("not found", str(ctx.exception).lower())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

