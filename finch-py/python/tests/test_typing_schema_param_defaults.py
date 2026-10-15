import unittest


class TypingSchemaParamDefaultsTest(unittest.TestCase):
    def test_enum_upper_snake_aliases_exist(self) -> None:
        import finch_db

        self.assertIs(finch_db.DataType.INT64, finch_db.DataType.Int64)
        self.assertIs(finch_db.MetricType.IP, finch_db.MetricType.InnerProduct)
        self.assertIs(finch_db.QuantizeType.FP16, finch_db.QuantizeType.Fp16)

    def test_invert_index_param_defaults(self) -> None:
        import finch_db

        p = finch_db.InvertIndexParam()
        self.assertFalse(bool(p.enable_range_optimization))
        self.assertFalse(bool(p.enable_extended_wildcard))

    def test_field_schema_nullable_default_false(self) -> None:
        import finch_db

        f = finch_db.FieldSchema("id", finch_db.DataType.INT64)
        self.assertFalse(bool(f.nullable))

    def test_collection_schema_duplicate_names_rejected(self) -> None:
        import finch_db

        schema = finch_db.CollectionSchema("dup")
        schema.add_field(finch_db.FieldSchema("x", finch_db.DataType.INT32))
        schema.add_field(finch_db.VectorSchema("v", dim=4, data_type=finch_db.DataType.VECTOR_FP32))
        with self.assertRaises(Exception):
            schema.add_field(finch_db.FieldSchema("x", finch_db.DataType.INT32))


if __name__ == "__main__":
    unittest.main(verbosity=2)

