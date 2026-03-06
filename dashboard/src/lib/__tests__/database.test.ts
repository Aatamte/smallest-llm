import { describe, it, expect, beforeEach, afterEach } from "vitest";
import initSqlJs from "sql.js";
import { Database } from "../database";
import type { TableSchema } from "../types";

let database: Database;

const usersSchema: TableSchema = {
  name: "users",
  columns: [
    { name: "id", type: "INTEGER", primaryKey: true },
    { name: "name", type: "TEXT", notNull: true },
  ],
};

const postsSchema: TableSchema = {
  name: "posts",
  columns: [
    { name: "id", type: "INTEGER", primaryKey: true },
    { name: "title", type: "TEXT", notNull: true },
    { name: "score", type: "REAL" },
  ],
  indexes: [{ name: "idx_posts_title", columns: ["title"] }],
};

beforeEach(async () => {
  const SQL = await initSqlJs();
  database = new Database();
  await database.init(new SQL.Database());
});

afterEach(() => {
  database.close();
});

describe("init", () => {
  it("creates a working sql.js database", () => {
    database.exec("CREATE TABLE test (id INTEGER PRIMARY KEY)");
    database.exec("INSERT INTO test VALUES (1)");
    const rows = database.query("SELECT * FROM test");
    expect(rows.length).toBe(1);
  });
});

describe("addTable", () => {
  it("creates table and returns Table instance", () => {
    const table = database.addTable(usersSchema);
    expect(table).toBeDefined();
    expect(table.schema.name).toBe("users");
  });

  it("table exists in DB after addTable", () => {
    database.addTable(usersSchema);
    const table = database.getTable("users");
    table.upsert({ id: 1, name: "alice" });
    expect(table.select().length).toBe(1);
  });
});

describe("getTable", () => {
  it("returns the table", () => {
    database.addTable(usersSchema);
    const table = database.getTable("users");
    expect(table.schema.name).toBe("users");
  });

  it("throws for unknown name", () => {
    expect(() => database.getTable("nope")).toThrow('Table "nope" not found');
  });
});

describe("multiple tables", () => {
  it("can coexist", () => {
    const users = database.addTable(usersSchema);
    const posts = database.addTable(postsSchema);
    users.upsert({ id: 1, name: "alice" });
    posts.upsert({ id: 1, title: "hello", score: 4.5 });
    expect(users.select().length).toBe(1);
    expect(posts.select().length).toBe(1);
  });
});

describe("close", () => {
  it("cleans up", () => {
    database.addTable(usersSchema);
    database.close();
    expect(database.tables.size).toBe(0);
  });
});

describe("exec", () => {
  it("runs arbitrary SQL", () => {
    database.addTable(usersSchema);
    database.exec("INSERT INTO users (id, name) VALUES (?, ?)", [1, "bob"]);
    const rows = database.getTable("users").select();
    expect(rows.length).toBe(1);
    expect(rows[0].name).toBe("bob");
  });
});

describe("query", () => {
  it("returns results from arbitrary SQL", () => {
    database.addTable(usersSchema);
    database.exec("INSERT INTO users (id, name) VALUES (1, 'a')");
    database.exec("INSERT INTO users (id, name) VALUES (2, 'b')");
    const rows = database.query<{ name: string }>("SELECT name FROM users ORDER BY name");
    expect(rows).toEqual([{ name: "a" }, { name: "b" }]);
  });

  it("returns empty array for no results", () => {
    database.addTable(usersSchema);
    const rows = database.query("SELECT * FROM users");
    expect(rows).toEqual([]);
  });
});

describe("getHashes", () => {
  it("returns hashes for all tables", () => {
    database.addTable(usersSchema);
    database.addTable(postsSchema);
    const hashes = database.getHashes();
    expect(hashes).toHaveProperty("users");
    expect(hashes).toHaveProperty("posts");
    expect(hashes.users).toBe(0);
    expect(hashes.posts).toBe(0);
  });

  it("hashes change after upserts", () => {
    const users = database.addTable(usersSchema);
    users.upsert({ id: 1, name: "alice" });
    const hashes = database.getHashes();
    expect(hashes.users).not.toBe(0);
  });
});

describe("applyOp", () => {
  it("routes upsert to the right table", () => {
    database.addTable(usersSchema);
    database.applyOp({ table: "users", op: "upsert", row: { id: 1, name: "alice" } });
    const rows = database.getTable("users").select();
    expect(rows.length).toBe(1);
    expect(rows[0].name).toBe("alice");
  });

  it("routes delete to the right table", () => {
    database.addTable(usersSchema);
    database.getTable("users").upsert({ id: 1, name: "alice" });
    database.applyOp({ table: "users", op: "delete", key: 1 });
    expect(database.getTable("users").select().length).toBe(0);
  });

  it("routes clear to the right table", () => {
    database.addTable(usersSchema);
    database.getTable("users").upsert({ id: 1, name: "a" });
    database.getTable("users").upsert({ id: 2, name: "b" });
    database.applyOp({ table: "users", op: "clear" });
    expect(database.getTable("users").select().length).toBe(0);
  });

  it("throws for unknown table", () => {
    expect(() => database.applyOp({ table: "nope", op: "clear" })).toThrow('Table "nope" not found');
  });

  it("throws for upsert without row", () => {
    database.addTable(usersSchema);
    expect(() => database.applyOp({ table: "users", op: "upsert" })).toThrow("upsert op requires row");
  });

  it("throws for delete without key", () => {
    database.addTable(usersSchema);
    expect(() => database.applyOp({ table: "users", op: "delete" })).toThrow("delete op requires key");
  });
});

describe("applyDump", () => {
  it("clears and loads rows", () => {
    const users = database.addTable(usersSchema);
    users.upsert({ id: 99, name: "old" });
    database.applyDump("users", [
      { id: 1, name: "a" },
      { id: 2, name: "b" },
      { id: 3, name: "c" },
    ]);
    const rows = users.select({ orderBy: "id" });
    expect(rows.length).toBe(3);
    expect(rows[0].name).toBe("a");
    expect(rows[1].name).toBe("b");
    expect(rows[2].name).toBe("c");
  });

  it("hash matches after dump", () => {
    database.addTable(usersSchema);
    database.applyDump("users", [
      { id: 1, name: "a" },
      { id: 2, name: "b" },
    ]);
    const hash = database.getHashes().users;
    expect(hash).not.toBe(0);
  });

  it("throws for unknown table", () => {
    expect(() => database.applyDump("nope", [])).toThrow('Table "nope" not found');
  });
});
