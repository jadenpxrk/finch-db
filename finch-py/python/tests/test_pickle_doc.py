from __future__ import annotations

import pickle
import shutil
import tempfile
import unittest

import finch_db


def _typed_fields() -> list[tuple[str, object, object]]:
    return [
        ("author", finch_db.FieldSchema("author", finch_db.DataType.STRING), "Tom"),
        ("blob", finch_db.FieldSchema("blob", finch_db.DataType.BINARY), b"\x00\xff"),
        ("tags", finch_db.FieldSchema("tags", finch_db.DataType.ARRAY_STRING), ["tag1", "tag2"]),
        ("weights", finch_db.FieldSchema("weights", finch_db.DataType.ARRAY_FLOAT), [1.5, 2.5]),
        ("chunks", finch_db.FieldSchema("chunks", finch_db.DataType.ARRAY_BINARY), [b"a", b"bc"]),
        ("dense16", finch_db.VectorSchema("dense16", dim=3, data_type=finch_db.DataType.VECTOR_FP16), [1.0, 2.0, 3.0]),
        ("dense32", finch_db.VectorSchema("dense32", dim=3, data_type=finch_db.DataType.VECTOR_FP32), [1.25, 2.25, 3.25]),
        ("s16", finch_db.VectorSchema("s16", data_type=finch_db.DataType.SPARSE_VECTOR_FP16), {1: 1.1, 2: 2.2}),
        ("s32", finch_db.VectorSchema("s32", data_type=finch_db.DataType.SPARSE_VECTOR_FP32), {1: 1.125, 2: 2.25}),
    ]


class PickleDocTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_core_doc_pickle_roundtrip_preserves_types_for_insert(self) -> None:
        from finch_db._finch import PyDoc as _Doc
        from finch_db._finch import create_and_open as _create_and_open

        root = tempfile.mkdtemp(prefix="finch_test_pickle_doc_")
        path = f"{root}/collection"
        try:
            schema = finch_db.CollectionSchema("pickle_doc")
            for _, field, _ in _typed_fields():
                schema.add_field(field)

            col = _create_and_open(
                path,
                schema._get_object(),
                finch_db.CollectionOption(read_only=False, enable_mmap=True),
            )

            d = _Doc()
            d.set_pk("p1")
            for name, field, value in _typed_fields():
                d.set_any(name, field._get_object(), value)

            d2 = pickle.loads(pickle.dumps(d))
            self.assertEqual(d2.pk(), "p1")
            self.assertEqual(d2.get_any("author", finch_db.DataType.STRING), "Tom")
            self.assertEqual(d2.get_any("blob", finch_db.DataType.BINARY), b"\x00\xff")
            self.assertEqual(d2.get_any("tags", finch_db.DataType.ARRAY_STRING), ["tag1", "tag2"])
            self.assertEqual(d2.get_any("chunks", finch_db.DataType.ARRAY_BINARY), [b"a", b"bc"])

            results = col.insert([d2])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].is_ok())

            got = col.fetch(["p1"])["p1"]
            self.assertEqual(got.get_any("author", finch_db.DataType.STRING), "Tom")
            self.assertEqual(got.get_any("blob", finch_db.DataType.BINARY), b"\x00\xff")
            self.assertEqual(got.get_any("tags", finch_db.DataType.ARRAY_STRING), ["tag1", "tag2"])
            self.assertEqual(got.get_any("weights", finch_db.DataType.ARRAY_FLOAT), [1.5, 2.5])
            self.assertEqual(got.get_any("chunks", finch_db.DataType.ARRAY_BINARY), [b"a", b"bc"])
            self.assertEqual(got.get_any("dense16", finch_db.DataType.VECTOR_FP16), [1.0, 2.0, 3.0])
            self.assertEqual(got.get_any("dense32", finch_db.DataType.VECTOR_FP32), [1.25, 2.25, 3.25])
            self.assertIsInstance(got.get_any("s16", finch_db.DataType.SPARSE_VECTOR_FP16), dict)
            self.assertIsInstance(got.get_any("s32", finch_db.DataType.SPARSE_VECTOR_FP32), dict)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

