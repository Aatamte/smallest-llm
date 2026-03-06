import initSqlJs, { type Database as SqlJsDatabase } from "sql.js";
import { Table } from "./table";
import type { TableSchema } from "./types";

export class Database {
  private _db: SqlJsDatabase | null = null;
  private _tables: Map<string, Table> = new Map();
  private _dumpedTables: Set<string> = new Set();

  /** Load sql.js WASM and create an in-memory database.
   *  Optionally pass an existing sql.js Database instance (for testing). */
  async init(existingDb?: SqlJsDatabase): Promise<void> {
    if (existingDb) {
      this._db = existingDb;
      return;
    }
    const SQL = await initSqlJs({ locateFile: () => "/sql-wasm.wasm" });
    this._db = new SQL.Database();
  }

  /** Register a table from a schema definition. Creates it in the DB and returns the Table instance. */
  addTable(schema: TableSchema): Table {
    const db = this._getDb();
    const table = new Table(schema, db);
    table.create();
    this._tables.set(schema.name, table);
    return table;
  }

  /** Get a registered table by name. Throws if not found. */
  getTable(name: string): Table {
    const table = this._tables.get(name);
    if (!table) throw new Error(`Table "${name}" not found`);
    return table;
  }

  /** All registered tables. */
  get tables(): Map<string, Table> {
    return this._tables;
  }

  /** Run arbitrary SQL (mutations). */
  exec(sql: string, params?: unknown[]): void {
    this._getDb().run(sql, params as (string | number | null | Uint8Array)[] | undefined);
  }

  /** Run arbitrary SELECT and return row objects. */
  query<T = Record<string, unknown>>(sql: string, params?: unknown[]): T[] {
    const results = this._getDb().exec(sql, params as (string | number | null | Uint8Array)[] | undefined);
    if (results.length === 0) return [];
    const { columns, values } = results[0];
    return values.map((row) => {
      const obj: Record<string, unknown> = {};
      columns.forEach((col, i) => {
        obj[col] = row[i];
      });
      return obj as T;
    });
  }

  /** Close the database. */
  /** Returns { tableName: hash } for all registered tables. */
  getHashes(): Record<string, number> {
    const hashes: Record<string, number> = {};
    for (const [name, table] of this._tables) {
      hashes[name] = table.getHash();
    }
    return hashes;
  }

  /** Apply a single CDC operation to the appropriate table. */
  applyOp(op: { table: string; op: "upsert" | "delete" | "clear"; row?: Record<string, unknown>; key?: unknown }): void {
    const table = this.getTable(op.table);
    switch (op.op) {
      case "upsert":
        if (!op.row) throw new Error("upsert op requires row");
        table.upsert(op.row);
        break;
      case "delete":
        if (op.key === undefined) throw new Error("delete op requires key");
        table.delete(op.key);
        break;
      case "clear":
        table.clear();
        break;
    }
  }

  /** Apply a batch of CDC operations (same table, same op type) with one version bump. */
  applyOps(msg: { table: string; op: "upsert"; rows: Record<string, unknown>[] }): void {
    const table = this.getTable(msg.table);
    table.bulkInsert(msg.rows);
  }

  /** Load rows into a table. First call clears the table; subsequent calls append (for chunked dumps). */
  applyDump(tableName: string, rows: Record<string, unknown>[]): void {
    const table = this.getTable(tableName);
    if (!this._dumpedTables.has(tableName)) {
      table.clear();
      this._dumpedTables.add(tableName);
    }
    table.bulkInsert(rows);
  }

  /** Reset dump tracking (call after sync is complete so next reconnect re-clears). */
  resetDumpState(): void {
    this._dumpedTables.clear();
  }

  /** Clear all data from all tables. */
  clearAll(): void {
    for (const table of this._tables.values()) {
      table.clear();
    }
    this._dumpedTables.clear();
  }

  close(): void {
    this._db?.close();
    this._db = null;
    this._tables.clear();
  }

  private _getDb(): SqlJsDatabase {
    if (!this._db) throw new Error("Database not initialized — call init() first");
    return this._db;
  }
}
