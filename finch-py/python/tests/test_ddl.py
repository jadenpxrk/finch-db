import shutil
import tempfile
import unittest


class DdlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch_db

        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_column_ddl_only_allows_basic_numeric(self) -> None:
        import finch_db

        root = tempfile.mkdtemp(prefix="finch_py_ddl_")
        path = f"{root}/collection"
        try:
            schema = finch_db.CollectionSchema("ddl")
            schema.add_field(
                finch_db.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch_db.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch_db.FieldSchema("id", finch_db.DataType.INT64, nullable=False))
            schema.add_field(finch_db.FieldSchema("name", finch_db.DataType.STRING, nullable=True))
            col = finch_db.create_and_open(path, schema, finch_db.CollectionOption())

            col.add_column(finch_db.FieldSchema("age", finch_db.DataType.INT32, nullable=True), expression="100")

            with self.assertRaises(ValueError):
                col.add_column(finch_db.FieldSchema("full_name", finch_db.DataType.STRING, nullable=True), expression="'x'")

            with self.assertRaises(ValueError):
                col.drop_column("name")

            with self.assertRaises(ValueError):
                col.alter_column(old_name="name", new_name="full_name")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
