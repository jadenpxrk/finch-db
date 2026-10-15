import shutil
import tempfile
import unittest


class FieldNameValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch_db

        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_add_column_rejects_invalid_names(self) -> None:
        import finch_db

        root = tempfile.mkdtemp(prefix="finch_py_name_")
        path = f"{root}/collection"
        try:
            schema = finch_db.CollectionSchema("name_test")
            schema.add_field(
                finch_db.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch_db.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch_db.FieldSchema("id", finch_db.DataType.INT64, nullable=False))
            col = finch_db.create_and_open(path, schema, finch_db.CollectionOption())

            invalid = ["", " ", "a" * 33, "field name", "field.name", "field@name", "field/name"]
            for name in invalid:
                with self.assertRaises(Exception):
                    col.add_column(finch_db.FieldSchema(name, finch_db.DataType.INT32), expression="100")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_alter_column_rejects_invalid_new_names(self) -> None:
        import finch_db

        root = tempfile.mkdtemp(prefix="finch_py_name_")
        path = f"{root}/collection"
        try:
            schema = finch_db.CollectionSchema("name_test2")
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
            col.insert(finch_db.Doc(id="1", fields={"x": 1}))
            col.flush()

            invalid = ["", " ", "a" * 33, "field name", "field.name", "field@name", "field/name"]
            for name in invalid:
                with self.assertRaises(Exception):
                    col.alter_column(old_name="x", new_name=name)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
