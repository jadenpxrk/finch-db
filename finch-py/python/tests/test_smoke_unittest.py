import shutil
import tempfile
import unittest


class FinchSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch_db

        # init is a one-shot global initializer.
        try:
            finch_db.init()
        except RuntimeError:
            pass

    def test_roundtrip_create_insert_index_query_stats(self) -> None:
        import finch_db

        root = tempfile.mkdtemp(prefix="finch_py_smoke_")
        path = f"{root}/collection"
        try:
            schema = finch_db.CollectionSchema("smoke")
            inv = finch_db.InvertIndexParam(enable_range_optimization=True, enable_extended_wildcard=False)
            schema.add_field(finch_db.FieldSchema("id", finch_db.DataType.Int64, nullable=False).with_invert_index(inv))
            schema.add_field(finch_db.VectorSchema("emb", dim=4, data_type=finch_db.DataType.VectorFp32))

            col = finch_db.create_and_open(path, schema, finch_db.CollectionOption(read_only=False, enable_mmap=True))

            docs = []
            for i in range(100):
                d = finch_db.Doc(
                    id=str(i),
                    fields={"id": i},
                    vectors={"emb": [float(i), 0.0, 0.0, 0.0]},
                )
                docs.append(d)
            results = col.insert(docs)
            self.assertEqual(len(results), 100)
            self.assertTrue(all(s.is_ok() for s in results))

            col.flush()

            col.create_hnsw_index(
                "emb",
                finch_db.HnswIndexParam(metric=finch_db.MetricType.L2, m=50, ef_construction=500),
                rebuild=True,
                option=finch_db.IndexOption(concurrency=0),
            )
            col.optimize(option=finch_db.OptimizeOption(concurrency=0))

            out = col.query(
                vectors=finch_db.VectorQuery(
                    "emb",
                    vector=[42.2, 0.0, 0.0, 0.0],
                    param=finch_db.HnswQueryParam(ef=300),
                ),
                topk=10,
                output_fields=[],
            )
            self.assertEqual(len(out), 10)
            self.assertEqual(out[0].id, "42")

            # Filter-only query.
            only = col.query(topk=5, filter="id >= 95", output_fields=["id"])
            self.assertTrue(all(int(d.id) >= 95 for d in only))

            s = col.stats
            d = s.to_dict()
            self.assertIn("doc_count", d)
            self.assertGreaterEqual(d["doc_count"], 100)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
