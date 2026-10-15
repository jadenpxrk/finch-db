export interface HnswIndexParams {
  metric: number
  m: number
  efConstruction: number
  quantize: number
}

export interface IvfIndexParams {
  metric: number
  nList: number
  nIters: number
  useSoar?: boolean
  quantize: number
}

export interface FlatIndexParams {
  metric: number
  quantize: number
}

export interface InvertIndexParams {
  enableRangeOptimization?: boolean
  enableExtendedWildcard?: boolean
}

export interface FieldSchemaOptions {
  name: string
  dataType: number
  /** Defaults to false, as in Rust and Python. */
  nullable?: boolean
  dimension?: number
  hnswIndex?: HnswIndexParams
  ivfIndex?: IvfIndexParams
  flatIndex?: FlatIndexParams
  invertIndex?: InvertIndexParams
}

export interface CollectionSchemaOptions {
  name: string
  fields: FieldSchemaOptions[]
  maxDocCountPerSegment?: number
}

export interface DocObject {
  pk: string
  score?: number
  fields: Record<string, any>
}

export interface VectorQueryOptions {
  fieldName: string
  queryVector: number[]
  /**
   * Binary32 query vector (Hamming). When provided (non-empty), it takes
   * precedence over `queryVector`.
   */
  queryVectorU32?: number[]
  /**
   * Binary64 query vector (Hamming), encoded as decimal strings to avoid JS
   * number precision loss. When provided (non-empty), it takes precedence over
   * `queryVector`.
   */
  queryVectorU64?: string[]
  topk: number
  filter?: string
  ef?: number
  nProbe?: number
  concurrency?: number
  bfPks?: string[]
  radius?: number
  isLinear?: boolean
  useRefiner?: boolean
  refinerK?: number
  refinerScaleFactor?: number
  sparseIndices?: number[]
  sparseValues?: number[]
  includeVector?: boolean
  includeDocId?: boolean
  outputFields?: string[]
}

export interface GroupResultObject {
  groupValue: any
  docs: DocObject[]
}

export interface StatusObject {
  ok: boolean
  code: number
  message: string
}

/** Errors from finch operations carry the numeric finch status code, as in `StatusObject.code`. */
export interface FinchError extends Error {
  code: number
}

export declare class Collection {
  insert(docs: DocObject[]): Promise<StatusObject[]>
  upsert(docs: DocObject[]): Promise<StatusObject[]>
  update(docs: DocObject[]): Promise<StatusObject[]>
  delete(pks: string[]): Promise<StatusObject[]>
  deleteByFilter(filter: string): Promise<StatusObject>
	  query(opts: VectorQueryOptions): DocObject[]
	  /**
	   * Execute a SQL SELECT query (Finch extension).
	   *
	   * Note: 64-bit unsigned integers (e.g. `_finch_g_doc_id_`, `_finch_row_id_`) are
	   * returned as decimal strings to avoid JavaScript precision loss.
	   */
	  querySql(sql: string): DocObject[]
  fetch(pks: string[]): Record<string, DocObject>
  statsInfo(): any
  groupByQuery(
    opts: VectorQueryOptions,
    groupByField: string,
    groupCount: number,
    groupTopk: number
  ): GroupResultObject[]
  createIndex(
    field: string,
    fieldOpts: FieldSchemaOptions,
    rebuild?: boolean,
    concurrency?: number
  ): Promise<void>
  dropIndex(field: string): Promise<void>
  addColumn(
    fieldOpts: FieldSchemaOptions,
    rebuildIndex?: boolean,
    concurrency?: number,
    expression?: string
  ): Promise<void>
  dropColumn(field: string): Promise<void>
  alterColumn(
    field: string,
    renameTo?: string,
    fieldSchema?: FieldSchemaOptions,
    rebuildIndex?: boolean,
    concurrency?: number
  ): Promise<void>
  optimize(maxSegments?: number, concurrency?: number): Promise<void>
	flush(): Promise<void>
	stats(): Record<string, number>
	schemaInfo(): any
	path(): string
	options(): any
	destroy(): void
}

export declare function createAndOpen(
  path: string,
  schema: CollectionSchemaOptions,
  readOnly?: boolean,
  enableMmap?: boolean,
  maxBufferSize?: number
): Promise<Collection>

export declare function open(
  path: string,
  readOnly?: boolean,
  enableMmap?: boolean,
  maxBufferSize?: number
): Promise<Collection>

export interface GlobalConfigOptions {
  memoryLimitBytes?: number
  /** 0=Debug, 1=Info, 2=Warn, 3=Error, 4=Fatal */
  logLevel?: number
  logToFile?: boolean
  logDir?: string
  logBasename?: string
  logFileSizeMb?: number
  logOverdueDays?: number
  queryThreadCount?: number
  walFlushEveryDocs?: number
  walFsyncEveryDocs?: number
  invertToForwardScanRatio?: number
  bruteForceByKeysRatio?: number
  optimizeThreadCount?: number
}

export declare function initGlobalConfig(opts?: GlobalConfigOptions): void

export declare enum MetricType {
  Undefined = 0,
  L2 = 1,
  InnerProduct = 2,
  Cosine = 3,
  MipsL2 = 4,
  Hamming = 5,
}

export declare enum DataType {
  Undefined = 0,
  Binary = 1,
  String = 2,
  Bool = 3,
  Int32 = 4,
  Int64 = 5,
  Uint32 = 6,
  Uint64 = 7,
  Float32 = 8,
  Float64 = 9,
  Int8 = 100,
  Int16 = 101,
  Uint8 = 102,
  Uint16 = 103,
  Float16 = 104,
  Bytes = 105,
  VectorBinary32 = 20,
  VectorBinary64 = 21,
  VectorFp16 = 22,
  VectorFp32 = 23,
  VectorFp64 = 24,
  VectorInt4 = 25,
  VectorInt8 = 26,
  VectorInt16 = 27,
  VectorBool = 120,
  VectorInt32 = 121,
  VectorInt64 = 122,
  VectorUint32 = 123,
  VectorUint64 = 124,
  SparseFp16 = 30,
  SparseFp32 = 31,
  ArrayBinary = 40,
  ArrayString = 41,
  ArrayBool = 42,
  ArrayInt32 = 43,
  ArrayInt64 = 44,
  ArrayUint32 = 45,
  ArrayUint64 = 46,
  ArrayFp32 = 47,
  ArrayFp64 = 48,
}

export declare enum QuantizeType {
  Undefined = 0,
  Fp16 = 1,
  Int8 = 2,
  Int4 = 3,
}
