# finch-db

finch-db is a vector database written in Rust. It runs as a library inside the calling
process.

It does these things:

- It stores documents with dense or sparse vectors and scalar fields in a directory on disk.
- It answers nearest-neighbor queries with HNSW, IVF, or flat indexes.
- It filters on scalar fields with a SQL-like syntax.
- After a crash, it recovers its last committed state from a write-ahead log and a manifest
  file.

## Layout

| Path | What it is |
|---|---|
| `crates/finch-types` | Schema, query, and error types |
| `crates/finch-proto` | Manifest encoding |
| `crates/finch-storage` | Columnar forward store (Arrow IPC or Parquet, optional mmap) |
| `crates/finch-core` | Index algorithms and quantization |
| `crates/finch-db` | Collections, segments, WAL, SQL-like filtering |
| `finch-py`, `finch-node` | Python and Node.js bindings |

## Rust

```toml
[dependencies]
finch-db = { path = "crates/finch-db" }
finch-types = { path = "crates/finch-types" }
```

```rust
use std::path::Path;

use finch_db::Collection;
use finch_types::{CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, VectorQuery};

let schema = CollectionSchema::new("notes")
    .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
let col = Collection::create_and_open(Path::new("./notes"), schema, CollectionOptions::default())?;

// `insert` returns one `Status` per document; a failed document does not fail the call.
col.insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])?;
col.flush()?;

let hits = col.query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1))?;
assert_eq!(hits[0].pk, "d1");
```

A field is required by default: `FieldSchema::new` in Rust, `FieldSchema` in Python, and
`FieldSchemaOptions.nullable` in Node all default to not nullable. Call `.nullable()` in Rust or
pass `nullable=True` to let a document omit the field or store null in it.

Run the tests:

```bash
cargo test --workspace
```

## Python

```bash
cd finch-py
python -m maturin develop --release
```

```python
import finch_db

schema = finch_db.CollectionSchema(
    name="notes",
    vectors=[finch_db.VectorSchema("emb", finch_db.DataType.VECTOR_FP32, dimension=4)],
)
col = finch_db.create_and_open("./notes", schema)

col.insert(finch_db.Doc(id="d1", vectors={"emb": [1.0, 0.0, 0.0, 0.0]}))
col.flush()

hits = col.query(finch_db.VectorQuery("emb", vector=[1.0, 0.0, 0.0, 0.0]), topk=1)
assert hits[0].id == "d1"
```

The Python API follows the Rust one. Its definitions are in `finch-py/python/finch_db`.

## Documentation

- `docs/architecture.md`: the crates, the write and read paths, and crash recovery.
- `docs/vectordbbench.md`: how to run VectorDBBench against finch-db.
- `docs/global-config.md`: process-wide settings.

## Building for one machine

On x86_64, building with `RUSTFLAGS="-C target-cpu=native"` lets the compiler use every
instruction set the build machine has. Use that flag only for a build that runs on the same
machine. A binary built with the flag can stop with an illegal-instruction error on a CPU
that lacks one of those instruction sets. Without the flag, the distance functions in
`finch-core` still choose AVX-512, AVX2, or SSE4.1 at runtime. The flag therefore changes
only code that the compiler vectorizes by itself.

## Known limitations

Each collection keeps its primary-key map in fjall, and fjall releases 3.0 to 3.1.10 can block
forever while a database shuts down. The report is
[fjall issue 260](https://github.com/fjall-rs/fjall/issues/260) and the fix is
[fjall pull request 321](https://github.com/fjall-rs/fjall/pull/321), merged but not yet
released. finch-db builds fjall from that merged commit through a source override in `Cargo.toml`
until a release contains it.

## Status

The crates are `finch-types`, `finch-proto`, `finch-storage`, `finch-core`, and `finch-db`.
They have 401 tests. One test carries the ignore attribute. All tests use synthetic data.

The project has not had a public release.

## License

AGPL-3.0-only. See `LICENSE`.
