from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import finch_db


class CreateOpenTest(unittest.TestCase):
    def _schema(self) -> finch_db.CollectionSchema:
        return finch_db.CollectionSchema(
            name="test_collection",
            fields=finch_db.FieldSchema("id", finch_db.DataType.INT64, nullable=False),
            vectors=finch_db.VectorSchema("image", finch_db.DataType.VECTOR_FP32, 4),
        )

    def test_create_and_open_valid_paths(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "a" / "b" / "col")
            schema = self._schema()
            opt = finch_db.CollectionOption(read_only=False, enable_mmap=True)
            col = finch_db.create_and_open(path=path, schema=schema, option=opt)
            self.assertEqual(col.path, path)
            self.assertEqual(col.schema.name, schema.name)
            self.assertEqual(list(col.schema.fields), list(schema.fields))
            self.assertEqual(list(col.schema.vectors), list(schema.vectors))
            self.assertEqual(col.option.read_only, opt.read_only)
            self.assertEqual(col.option.enable_mmap, opt.enable_mmap)
            col.destroy()

    def test_create_and_open_valid_options(self):
        with tempfile.TemporaryDirectory() as td:
            schema = self._schema()
            path1 = str(Path(td) / "col1")
            col1 = finch_db.create_and_open(path=path1, schema=schema, option=finch_db.CollectionOption(read_only=False, enable_mmap=True))
            col1.destroy()

            path2 = str(Path(td) / "col2")
            col2 = finch_db.create_and_open(path=path2, schema=schema, option=finch_db.CollectionOption(read_only=False, enable_mmap=False))
            col2.destroy()

    def test_create_and_open_read_only_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "col")
            with self.assertRaises(ValueError):
                finch_db.create_and_open(path=path, schema=self._schema(), option=finch_db.CollectionOption(read_only=True, enable_mmap=True))

    def test_create_and_open_invalid_path(self):
        with self.assertRaises(ValueError):
            finch_db.create_and_open(path="", schema=self._schema())
        with self.assertRaises(ValueError):
            finch_db.create_and_open(path=" has_space", schema=self._schema())
        with self.assertRaises(ValueError):
            finch_db.create_and_open(path="has_space ", schema=self._schema())
        with self.assertRaises(ValueError):
            finch_db.create_and_open(path="invalid:path", schema=self._schema())
        with self.assertRaises(ValueError):
            finch_db.create_and_open(path="test@#$%collection", schema=self._schema())

    def test_create_and_open_relative_path_with_slashes(self):
        with tempfile.TemporaryDirectory() as td:
            prev = os.getcwd()
            os.chdir(td)
            try:
                rel = "test/collection/with/slashes"
                col = finch_db.create_and_open(path=rel, schema=self._schema())
                self.assertEqual(col.path, rel)
                col.destroy()
            finally:
                os.chdir(prev)

    def test_open_invalid_path(self):
        opt = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        with self.assertRaises(ValueError):
            finch_db.open("", opt)
        with self.assertRaises(ValueError):
            finch_db.open("has_space ", opt)

    def test_open_missing_collection_raises(self):
        with tempfile.TemporaryDirectory() as td:
            missing = str(Path(td) / "missing")
            with self.assertRaises(Exception):
                finch_db.open(missing, finch_db.CollectionOption(read_only=False, enable_mmap=True))
