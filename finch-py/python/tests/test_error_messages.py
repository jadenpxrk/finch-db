import shutil
import tempfile
import unittest


class ErrorMessagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch_db

        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_alter_column_missing_column_message(self) -> None:
        import finch_db

        root = tempfile.mkdtemp(prefix="finch_py_errmsg_")
        path = f"{root}/collection"
        try:
            schema = finch_db.CollectionSchema("errmsg")
            # A collection needs at least one vector field.
            schema.add_field(
                finch_db.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch_db.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch_db.FieldSchema("x", finch_db.DataType.INT32, nullable=True))
            col = finch_db.create_and_open(path, schema, finch_db.CollectionOption())
            with self.assertRaises(Exception) as ctx:
                col.alter_column(old_name="non_existing", new_name="y")
            self.assertIn("column non_existing not found", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
