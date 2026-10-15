from __future__ import annotations

import unittest

import finch_db


class TypingTest(unittest.TestCase):
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

    def test_members_exist(self):
        for m in [
            "STRING",
            "BOOL",
            "INT32",
            "INT64",
            "FLOAT",
            "DOUBLE",
            "UINT32",
            "UINT64",
            "VECTOR_FP16",
            "VECTOR_FP32",
            "VECTOR_FP64",
            "VECTOR_INT8",
            "SPARSE_VECTOR_FP16",
            "SPARSE_VECTOR_FP32",
            "ARRAY_STRING",
            "ARRAY_INT32",
            "ARRAY_INT64",
            "ARRAY_FLOAT",
            "ARRAY_DOUBLE",
            "ARRAY_BOOL",
            "ARRAY_UINT32",
            "ARRAY_UINT64",
        ]:
            self.assertIn(m, finch_db.DataType.__members__)

        for m in ["HNSW", "IVF", "FLAT", "INVERT"]:
            self.assertIn(m, finch_db.IndexType.__members__)

        for m in ["L2", "IP", "COSINE", "MIPS_L2", "HAMMING"]:
            self.assertIn(m, finch_db.MetricType.__members__)

        for m in ["FP16", "INT8", "INT4", "UNDEFINED"]:
            self.assertIn(m, finch_db.QuantizeType.__members__)

        for m in [
            "OK",
            "UNKNOWN",
            "NOT_FOUND",
            "ALREADY_EXISTS",
            "INVALID_ARGUMENT",
            "PERMISSION_DENIED",
            "FAILED_PRECONDITION",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL_ERROR",
            "NOT_SUPPORTED",
        ]:
            self.assertIn(m, finch_db.StatusCode.__members__)

    def test_status_api(self):
        s = finch_db.Status()
        self.assertTrue(s.ok())
        self.assertEqual(s.code(), finch_db.StatusCode.OK)
        self.assertEqual(s.message(), "")
        self.assertTrue(bool(s))

        s2 = finch_db.Status(finch_db.StatusCode.INVALID_ARGUMENT, "bad")
        self.assertFalse(s2.ok())
        self.assertEqual(s2.code(), finch_db.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(s2.message(), "bad")
        # A Status is truthy for every code, so `bool(result)` only checks for None.
        self.assertTrue(bool(s2))

        self.assertTrue(finch_db.Status.OK().ok())
        self.assertEqual(finch_db.Status.NotFound("x").code(), finch_db.StatusCode.NOT_FOUND)

        # Some factories map to the closest Finch status code.
        self.assertEqual(finch_db.Status.AlreadyExists("x").code(), finch_db.StatusCode.ALREADY_EXISTS)
        self.assertEqual(finch_db.Status.InvalidArgument("x").code(), finch_db.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(finch_db.Status.PermissionDenied("x").code(), finch_db.StatusCode.PERMISSION_DENIED)
        self.assertIsNotNone(finch_db.Status.FailedPrecondition("x"))
        self.assertIsNotNone(finch_db.Status.ResourceExhausted("x"))
        self.assertIsNotNone(finch_db.Status.Unavailable("x"))
        self.assertIsNotNone(finch_db.Status.InternalError("x"))
        self.assertIsNotNone(finch_db.Status.NotSupported("x"))
        self.assertIsNotNone(finch_db.Status.Unknown("x"))
