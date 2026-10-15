from __future__ import annotations

import unittest

import finch_db


class ParamTest(unittest.TestCase):
    def test_core_param_submodule_aliases(self):
        from finch_db._finch import _Doc  # type: ignore[attr-defined]
        from finch_db._finch.param import _VectorQuery  # type: ignore[attr-defined]

        d = _Doc()
        self.assertIsNotNone(d)

        v = _VectorQuery()
        self.assertIsNotNone(v)
        self.assertEqual(int(v.topk), 10)

    def test_invert_index_param(self):
        param = finch_db.InvertIndexParam()
        self.assertFalse(param.enable_range_optimization)
        self.assertFalse(param.enable_extended_wildcard)
        self.assertEqual(param.type, finch_db.IndexType.INVERT)

        param2 = finch_db.InvertIndexParam(enable_range_optimization=True, enable_extended_wildcard=True)
        self.assertTrue(param2.enable_range_optimization)
        self.assertTrue(param2.enable_extended_wildcard)

        with self.assertRaises(AttributeError):
            param.enable_range_optimization = True  # type: ignore[misc]

    def test_hnsw_index_param(self):
        param = finch_db.HnswIndexParam()
        self.assertEqual(param.metric_type, finch_db.MetricType.IP)
        self.assertEqual(param.m, 50)
        self.assertEqual(param.ef_construction, 500)
        self.assertEqual(param.quantize_type, finch_db.QuantizeType.UNDEFINED)
        self.assertEqual(param.type, finch_db.IndexType.HNSW)
        self.assertFalse(hasattr(param, "scaling_factor"))

        param2 = finch_db.HnswIndexParam(metric_type=finch_db.MetricType.L2, m=10, ef_construction=1000, quantize_type=finch_db.QuantizeType.FP16)
        self.assertEqual(param2.metric_type, finch_db.MetricType.L2)
        self.assertEqual(param2.m, 10)
        self.assertEqual(param2.ef_construction, 1000)
        self.assertEqual(param2.quantize_type, finch_db.QuantizeType.FP16)

        with self.assertRaises(TypeError):
            finch_db.HnswIndexParam(scaling_factor=50)  # type: ignore[call-arg]

        with self.assertRaises(AttributeError):
            param.m = 1  # type: ignore[misc]

    def test_flat_index_param(self):
        param = finch_db.FlatIndexParam()
        self.assertEqual(param.type, finch_db.IndexType.FLAT)
        self.assertEqual(param.quantize_type, finch_db.QuantizeType.UNDEFINED)
        self.assertEqual(param.metric_type, finch_db.MetricType.IP)
        self.assertFalse(hasattr(param, "column_major"))

        param2 = finch_db.FlatIndexParam(metric_type=finch_db.MetricType.L2, quantize_type=finch_db.QuantizeType.INT8)
        self.assertEqual(param2.metric_type, finch_db.MetricType.L2)
        self.assertEqual(param2.quantize_type, finch_db.QuantizeType.INT8)

        with self.assertRaises(AttributeError):
            param.quantize_type = finch_db.QuantizeType.INT4  # type: ignore[misc]

    def test_ivf_index_param(self):
        param = finch_db.IVFIndexParam()
        self.assertEqual(param.metric_type, finch_db.MetricType.IP)
        self.assertEqual(param.n_list, 0)
        self.assertEqual(param.quantize_type, finch_db.QuantizeType.UNDEFINED)
        self.assertEqual(param.type, finch_db.IndexType.IVF)
        self.assertFalse(hasattr(param, "l1_index"))

        param2 = finch_db.IVFIndexParam(metric_type=finch_db.MetricType.L2, n_list=1000, quantize_type=finch_db.QuantizeType.FP16)
        self.assertEqual(param2.metric_type, finch_db.MetricType.L2)
        self.assertEqual(param2.n_list, 1000)
        self.assertEqual(param2.quantize_type, finch_db.QuantizeType.FP16)
        self.assertEqual(param2.type, finch_db.IndexType.IVF)

        with self.assertRaises(TypeError):
            finch_db.IVFIndexParam(l1_index=finch_db.IVFIndexParam())  # type: ignore[call-arg]

        with self.assertRaises(AttributeError):
            param.n_list = 10  # type: ignore[misc]

    def test_options(self):
        option = finch_db.CollectionOption()
        self.assertFalse(option.read_only)
        self.assertTrue(option.enable_mmap)
        with self.assertRaises(AttributeError):
            option.read_only = True  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            option.enable_mmap = False  # type: ignore[misc]

        index_opt = finch_db.IndexOption()
        self.assertEqual(index_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            index_opt.concurrency = 1  # type: ignore[misc]

        opt_opt = finch_db.OptimizeOption()
        self.assertEqual(opt_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            opt_opt.concurrency = 1  # type: ignore[misc]

        add_opt = finch_db.AddColumnOption()
        self.assertEqual(add_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            add_opt.concurrency = 1  # type: ignore[misc]

        alter_opt = finch_db.AlterColumnOption()
        self.assertEqual(alter_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            alter_opt.concurrency = 1  # type: ignore[misc]

    def test_hnsw_query_param(self):
        p = finch_db.HnswQueryParam()
        self.assertEqual(p.ef, 300)
        self.assertEqual(p.radius, 0.0)
        self.assertFalse(p.is_using_refiner)
        self.assertFalse(p.is_linear)

        p2 = finch_db.HnswQueryParam(ef=10, is_using_refiner=True, radius=30, is_linear=True)
        self.assertEqual(p2.ef, 10)
        self.assertEqual(p2.radius, 30.0)
        self.assertTrue(p2.is_using_refiner)
        self.assertTrue(p2.is_linear)

        with self.assertRaises(AttributeError):
            p.ef = 10  # type: ignore[misc]

    def test_ivf_query_param(self):
        p = finch_db.IVFQueryParam()
        self.assertEqual(p.nprobe, 10)
        self.assertEqual(p.n_probe, 10)

        with self.assertRaises(AttributeError):
            p.nprobe = 5  # type: ignore[misc]
