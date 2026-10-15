from __future__ import annotations

import re
import sys
import unittest

import finch_db


def _readonly_match_pattern() -> str:
    if sys.version_info >= (3, 11):
        return r"(can't set attribute|has no setter|readonly attribute)"
    return r"can't set attribute"


class ReadonlyErrorMessagesTest(unittest.TestCase):
    def test_field_schema_readonly(self):
        field = finch_db.FieldSchema(
            name="float",
            data_type=finch_db.DataType.FLOAT,
            nullable=True,
            index_param=finch_db.InvertIndexParam(),
        )
        with self.assertRaises(AttributeError) as ctx:
            field.index_param = finch_db.InvertIndexParam(enable_range_optimization=True)
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_invert_index_param_readonly(self):
        param = finch_db.InvertIndexParam()
        with self.assertRaises(AttributeError) as ctx:
            param.enable_range_optimization = True
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_collection_option_readonly(self):
        opt = finch_db.CollectionOption(read_only=False, enable_mmap=True)
        with self.assertRaises(AttributeError) as ctx:
            opt.read_only = True
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_query_param_readonly(self):
        param = finch_db.HnswQueryParam()
        with self.assertRaises(AttributeError) as ctx:
            param.ef = 10
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))
