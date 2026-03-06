import { describe, it, expect, beforeEach } from "vitest";
import initSqlJs, { type Database as SqlJsDatabase } from "sql.js";
import { Table } from "../table";
import type { TableSchema } from "../types";

let db: SqlJsDatabase;

beforeEach(async () => {
  const SQL = await initSqlJs();
  db = new SQL.Database();
});

const usersSchema: TableSchema = {
  name: "users",
  columns: [
    { name: "id", type: "INTEGER", primaryKey: true },
    { name: "name", type: "TEXT", notNull: true },
    { name: "email", type: "TEXT" },
    { name: "score", type: "REAL" },
  ],
  indexes: [{ name: "idx_users_name", columns: ["name"] }],
};

function makeTable(schema: TableSchema = usersSchema): Table {
  const t = new Table(schema, db);
  t.create();
  return t;
}

// ── DDL ──────────────────────────────────────────────────

describe("DDL", () => {
  it("creates a table with all column options", () => {
    const schema: TableSchema = {
      name: "items",
      columns: [
        { name: "id", type: "INTEGER", primaryKey: true },
        { name: "label", type: "TEXT", notNull: true },
        { name: "created_at", type: "TEXT", default: "datetime('now')" },
        { name: "value", type: "REAL" },
      ],
    };
    const t = makeTable(schema);
    t.upsert({ id: 1, label: "test" });
    expect(t.select().length).toBe(1);
  });

  it("creates indexes", () => {
    makeTable();
    const result = db.exec("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_users_name'");
    expect(result.length).toBe(1);
    expect(result[0].values[0][0]).toBe("idx_users_name");
  });

  it("create is idempotent", () => {
    const t = makeTable();
    expect(() => t.create()).not.toThrow();
  });

  it("throws if no primary key column", () => {
    const schema: TableSchema = {
      name: "bad",
      columns: [
        { name: "id", type: "INTEGER" },
        { name: "name", type: "TEXT" },
      ],
    };
    expect(() => new Table(schema, db)).toThrow("must have exactly one primaryKey column, got 0");
  });

  it("throws if multiple primary key columns", () => {
    const schema: TableSchema = {
      name: "bad",
      columns: [
        { name: "id", type: "INTEGER", primaryKey: true },
        { name: "name", type: "TEXT", primaryKey: true },
      ],
    };
    expect(() => new Table(schema, db)).toThrow("must have exactly one primaryKey column, got 2");
  });

  it("exposes pkCol", () => {
    const t = makeTable();
    expect(t.pkCol).toBe("id");
  });
});

// ── upsert ──────────────────────────────────────────────

describe("upsert", () => {
  it("inserts a row", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "alice", email: "a@b.com", score: 9.5 });
    const rows = t.select();
    expect(rows.length).toBe(1);
    expect(rows[0].name).toBe("alice");
  });

  it("inserts with subset of columns", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "bob" });
    const rows = t.select();
    expect(rows[0].email).toBeNull();
    expect(rows[0].score).toBeNull();
  });

  it("replaces existing row with same PK", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "alice", score: 1 });
    t.upsert({ id: 1, name: "alice-updated", score: 99 });
    const rows = t.select();
    expect(rows.length).toBe(1);
    expect(rows[0].name).toBe("alice-updated");
    expect(rows[0].score).toBe(99);
  });

  it("upsert is idempotent — same data gives same result", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "alice", score: 5 });
    t.upsert({ id: 1, name: "alice", score: 5 });
    const rows = t.select();
    expect(rows.length).toBe(1);
  });

  it("throws if PK is missing", () => {
    const t = makeTable();
    expect(() => t.upsert({ name: "alice" })).toThrow('must include primary key column "id"');
  });

  it("throws if PK is null", () => {
    const t = makeTable();
    expect(() => t.upsert({ id: null, name: "alice" })).toThrow('must include primary key column "id"');
  });
});

// ── upsertMany ──────────────────────────────────────────

describe("upsertMany", () => {
  it("batch upserts multiple rows", () => {
    const t = makeTable();
    t.upsertMany([
      { id: 1, name: "a", score: 1 },
      { id: 2, name: "b", score: 2 },
      { id: 3, name: "c", score: 3 },
    ]);
    expect(t.select().length).toBe(3);
  });

  it("empty array is a no-op", () => {
    const t = makeTable();
    t.upsertMany([]);
    expect(t.select()).toEqual([]);
  });

  it("replaces existing rows by PK", () => {
    const t = makeTable();
    t.upsertMany([
      { id: 1, name: "a", score: 1 },
      { id: 2, name: "b", score: 2 },
    ]);
    t.upsertMany([
      { id: 1, name: "a-updated", score: 10 },
      { id: 2, name: "b-updated", score: 20 },
    ]);
    const rows = t.select({ orderBy: "id" });
    expect(rows.length).toBe(2);
    expect(rows[0].name).toBe("a-updated");
    expect(rows[1].name).toBe("b-updated");
  });

  it("throws if any row missing PK", () => {
    const t = makeTable();
    expect(() => t.upsertMany([{ id: 1, name: "a" }, { name: "b" }])).toThrow(
      'must include primary key column "id"'
    );
  });
});

// ── select ──────────────────────────────────────────────

describe("select", () => {
  it("returns all rows when no filters", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    t.upsert({ id: 2, name: "b" });
    expect(t.select().length).toBe(2);
  });

  it("filters with WHERE + params", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a", score: 1 });
    t.upsert({ id: 2, name: "b", score: 2 });
    const rows = t.select({ where: "name = ?", params: ["a"] });
    expect(rows.length).toBe(1);
    expect(rows[0].name).toBe("a");
  });

  it("orders by orderBy", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "b", score: 2 });
    t.upsert({ id: 2, name: "a", score: 1 });
    const rows = t.select({ orderBy: "score ASC" });
    expect(rows[0].name).toBe("a");
    expect(rows[1].name).toBe("b");
  });

  it("selects specific columns", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a", email: "x@y.com" });
    const rows = t.select({ columns: "name" });
    expect(rows[0]).toEqual({ name: "a" });
    expect(rows[0]).not.toHaveProperty("email");
  });

  it("returns empty array for no matches", () => {
    const t = makeTable();
    const rows = t.select({ where: "name = ?", params: ["nope"] });
    expect(rows).toEqual([]);
  });
});

// ── delete ──────────────────────────────────────────────

describe("delete", () => {
  it("deletes by primary key", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    t.upsert({ id: 2, name: "b" });
    t.delete(1);
    const rows = t.select();
    expect(rows.length).toBe(1);
    expect(rows[0].name).toBe("b");
  });

  it("no-op for non-existent key", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    t.delete(999);
    expect(t.select().length).toBe(1);
  });
});

// ── clear ───────────────────────────────────────────────

describe("clear", () => {
  it("removes all rows", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    t.upsert({ id: 2, name: "b" });
    t.clear();
    expect(t.select()).toEqual([]);
  });

  it("table still works after clear", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    t.clear();
    t.upsert({ id: 2, name: "b" });
    expect(t.select().length).toBe(1);
  });
});


// ── Hashing ─────────────────────────────────────────────

describe("hashing", () => {
  it("hash starts at 0", () => {
    const t = makeTable();
    expect(t.getHash()).toBe(0);
  });

  it("hash changes after upsert", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    expect(t.getHash()).not.toBe(0);
  });

  it("hash returns to 0 after upsert then delete of same row", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    expect(t.getHash()).not.toBe(0);
    t.delete(1);
    expect(t.getHash()).toBe(0);
  });

  it("hash is 0 after clear", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a" });
    t.upsert({ id: 2, name: "b" });
    expect(t.getHash()).not.toBe(0);
    t.clear();
    expect(t.getHash()).toBe(0);
  });

  it("hash is same regardless of insert order (XOR is commutative)", () => {
    const t1 = makeTable();
    t1.upsert({ id: 1, name: "a", score: 1 });
    t1.upsert({ id: 2, name: "b", score: 2 });

    const schema2: TableSchema = {
      name: "users2",
      columns: usersSchema.columns,
      indexes: [],
    };
    const t2 = new Table(schema2, db);
    t2.create();
    t2.upsert({ id: 2, name: "b", score: 2 });
    t2.upsert({ id: 1, name: "a", score: 1 });

    expect(t1.getHash()).toBe(t2.getHash());
  });

  it("upsert same row twice gives same hash as inserting once", () => {
    const t1 = makeTable();
    t1.upsert({ id: 1, name: "a", score: 5 });
    const hashOnce = t1.getHash();

    const schema2: TableSchema = {
      name: "users3",
      columns: usersSchema.columns,
      indexes: [],
    };
    const t2 = new Table(schema2, db);
    t2.create();
    t2.upsert({ id: 1, name: "a", score: 5 });
    t2.upsert({ id: 1, name: "a", score: 5 });

    expect(hashOnce).toBe(t2.getHash());
  });

  it("different data gives different hash", () => {
    const t1 = makeTable();
    t1.upsert({ id: 1, name: "a" });

    const schema2: TableSchema = {
      name: "users4",
      columns: usersSchema.columns,
      indexes: [],
    };
    const t2 = new Table(schema2, db);
    t2.create();
    t2.upsert({ id: 1, name: "b" });

    expect(t1.getHash()).not.toBe(t2.getHash());
  });

  it("upsert updating a row changes the hash", () => {
    const t = makeTable();
    t.upsert({ id: 1, name: "a", score: 1 });
    const h1 = t.getHash();
    t.upsert({ id: 1, name: "a", score: 99 });
    const h2 = t.getHash();
    expect(h1).not.toBe(h2);
  });
});
