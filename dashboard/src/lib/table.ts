import { useSyncExternalStore } from "react";
import type { Database as SqlJsDatabase } from "sql.js";
import type { TableSchema } from "./types";

type Listener = () => void;

/** djb2 hash over a string → unsigned 32-bit integer. */
function djb2(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) & 0xffffffff;
  }
  return hash >>> 0;
}

export class Table {
  readonly schema: TableSchema;
  readonly pkCol: string;
  private _db: SqlJsDatabase;
  private _version = 0;
  private _hash = 0;
  private _listeners: Set<Listener> = new Set();
  private _colNames: string[];

  constructor(schema: TableSchema, db: SqlJsDatabase) {
    const pkCols = schema.columns.filter((c) => c.primaryKey);
    if (pkCols.length !== 1) {
      throw new Error(
        `Table "${schema.name}" must have exactly one primaryKey column, got ${pkCols.length}`
      );
    }
    this.schema = schema;
    this.pkCol = pkCols[0].name;
    this._db = db;
    this._colNames = schema.columns.map((c) => c.name);
  }

  /** Generate and execute CREATE TABLE + CREATE INDEX statements. */
  create(): void {
    const colDefs = this.schema.columns.map((c) => {
      const parts = [c.name, c.type];
      if (c.primaryKey) parts.push("PRIMARY KEY");
      if (c.autoincrement) parts.push("AUTOINCREMENT");
      if (c.notNull) parts.push("NOT NULL");
      if (c.default !== undefined) parts.push(`DEFAULT (${c.default})`);
      return parts.join(" ");
    });

    this._db.run(
      `CREATE TABLE IF NOT EXISTS ${this.schema.name} (\n  ${colDefs.join(",\n  ")}\n);`
    );

    for (const idx of this.schema.indexes ?? []) {
      this._db.run(
        `CREATE INDEX IF NOT EXISTS ${idx.name} ON ${this.schema.name}(${idx.columns.join(", ")});`
      );
    }
  }

  /** Upsert a row (INSERT OR REPLACE). Returns lastInsertRowid. */
  upsert(row: Record<string, unknown>): number {
    const pkVal = row[this.pkCol];
    if (pkVal === undefined || pkVal === null) {
      throw new Error(`Row must include primary key column "${this.pkCol}"`);
    }
    // XOR out old row hash if it exists
    const existing = this._getByPk(pkVal);
    if (existing) {
      this._hash ^= this._hashRow(existing);
    }

    const cols = Object.keys(row);
    const placeholders = cols.map(() => "?").join(", ");
    const values = cols.map((c) => row[c]);
    this._db.run(
      `INSERT OR REPLACE INTO ${this.schema.name} (${cols.join(", ")}) VALUES (${placeholders})`,
      values as (string | number | null | Uint8Array)[]
    );

    // XOR in new row hash — read back the full row to get defaults
    const inserted = this._getByPk(pkVal);
    if (inserted) {
      this._hash ^= this._hashRow(inserted);
    }

    const result = this._db.exec("SELECT last_insert_rowid() as id");
    this._bump();
    return result.length > 0 ? (result[0].values[0][0] as number) : 0;
  }

  /** Batch upsert multiple rows. Single version bump + hash update. */
  upsertMany(rows: Record<string, unknown>[]): void {
    if (rows.length === 0) return;

    for (const row of rows) {
      const pkVal = row[this.pkCol];
      if (pkVal === undefined || pkVal === null) {
        throw new Error(`Row must include primary key column "${this.pkCol}"`);
      }
      const existing = this._getByPk(pkVal);
      if (existing) {
        this._hash ^= this._hashRow(existing);
      }

      const cols = Object.keys(row);
      const placeholders = cols.map(() => "?").join(", ");
      const values = cols.map((c) => row[c]);
      this._db.run(
        `INSERT OR REPLACE INTO ${this.schema.name} (${cols.join(", ")}) VALUES (${placeholders})`,
        values as (string | number | null | Uint8Array)[]
      );

      const inserted = this._getByPk(pkVal);
      if (inserted) {
        this._hash ^= this._hashRow(inserted);
      }
    }

    this._bump();
  }

  /** Select rows with optional WHERE, params, orderBy, columns. */
  select(opts?: {
    where?: string;
    params?: unknown[];
    orderBy?: string;
    columns?: string;
  }): Record<string, unknown>[] {
    let sql = `SELECT ${opts?.columns ?? "*"} FROM ${this.schema.name}`;
    if (opts?.where) sql += ` WHERE ${opts.where}`;
    if (opts?.orderBy) sql += ` ORDER BY ${opts.orderBy}`;

    const results = this._db.exec(sql, opts?.params as (string | number | null | Uint8Array)[] | undefined);
    if (results.length === 0) return [];

    const { columns, values } = results[0];
    return values.map((row) => {
      const obj: Record<string, unknown> = {};
      columns.forEach((col, i) => {
        obj[col] = row[i];
      });
      return obj;
    });
  }

  /** Delete a row by primary key value. */
  delete(key: unknown): void {
    const existing = this._getByPk(key);
    if (existing) {
      this._hash ^= this._hashRow(existing);
    }
    this._db.run(
      `DELETE FROM ${this.schema.name} WHERE ${this.pkCol} = ?`,
      [key] as (string | number | null | Uint8Array)[]
    );
    this._bump();
  }

  /** Delete all rows. Resets hash to 0. */
  clear(): void {
    this._db.run(`DELETE FROM ${this.schema.name}`);
    this._hash = 0;
    this._bump();
  }

  /** Current content hash (XOR of all row hashes). */
  getHash(): number {
    return this._hash;
  }

  /** Current version number. Incremented on every mutation. */
  getVersion(): number {
    return this._version;
  }

  /** Subscribe to mutations. Returns unsubscribe function. */
  subscribe(callback: Listener): () => void {
    this._listeners.add(callback);
    return () => {
      this._listeners.delete(callback);
    };
  }

  /**
   * React hook: subscribe to this table and re-run queryFn on mutations.
   * Uses useSyncExternalStore — only re-renders when this table changes.
   */
  useQuery<T>(queryFn: () => T): T {
    useSyncExternalStore(
      (cb) => this.subscribe(cb),
      () => this._version,
    );
    return queryFn();
  }

  /** Hash a row's values using djb2. */
  private _hashRow(row: Record<string, unknown>): number {
    const sorted = this._colNames.slice().sort();
    const str = JSON.stringify(sorted.map((c) => row[c]));
    return djb2(str);
  }

  /** Look up a single row by primary key. Returns null if not found. */
  private _getByPk(key: unknown): Record<string, unknown> | null {
    const results = this._db.exec(
      `SELECT * FROM ${this.schema.name} WHERE ${this.pkCol} = ?`,
      [key] as (string | number | null | Uint8Array)[]
    );
    if (results.length === 0 || results[0].values.length === 0) return null;
    const { columns, values } = results[0];
    const obj: Record<string, unknown> = {};
    columns.forEach((col, i) => {
      obj[col] = values[0][i];
    });
    return obj;
  }

  private _bump(): void {
    this._version++;
    for (const listener of this._listeners) {
      listener();
    }
  }
}
