from __future__ import annotations

import unittest

import finch_db


class TypingEnumsAndStatusTest(unittest.TestCase):
    def test_enum_names_and_values(self):
        self.assertEqual(finch_db.DataType.FLOAT.name, "FLOAT")
        self.assertEqual(finch_db.DataType.FLOAT.value, 8)
        self.assertEqual(finch_db.IndexType.HNSW.name, "HNSW")
        self.assertEqual(finch_db.IndexType.HNSW.value, 1)
        self.assertEqual(finch_db.MetricType.COSINE.name, "COSINE")
        self.assertEqual(finch_db.MetricType.COSINE.value, 3)
        self.assertEqual(finch_db.QuantizeType.INT8.name, "INT8")
        self.assertEqual(finch_db.QuantizeType.INT8.value, 2)
        self.assertEqual(finch_db.StatusCode.OK.name, "OK")
        self.assertEqual(finch_db.StatusCode.OK.value, 0)

        for m in ["L2", "IP", "COSINE"]:
            self.assertIn(m, finch_db.MetricType.__members__)

        for m in ["HNSW", "IVF", "FLAT", "INVERT"]:
            self.assertIn(m, finch_db.IndexType.__members__)

        for m in ["FP16", "INT8", "INT4", "UNDEFINED"]:
            self.assertIn(m, finch_db.QuantizeType.__members__)

    def test_status_api(self):
        s = finch_db.Status(finch_db.StatusCode.OK)
        self.assertEqual(s.code(), finch_db.StatusCode.OK)
        self.assertTrue(s.ok())

        s2 = finch_db.Status(finch_db.StatusCode.NOT_FOUND, "Not Found")
        self.assertEqual(s2.message(), "Not Found")
        self.assertFalse(s2.ok())
