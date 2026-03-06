import { useMemo, useState } from "react";
import { db } from "../lib/db";
import { useTableQuery } from "../db/hooks";

const ALL_TABLES = ["runs", "metrics", "checkpoints", "models", "generations", "logs"];

export function TablesPage() {
  const [selectedTable, setSelectedTable] = useState<string | null>(null);

  const tableNames = useTableQuery(ALL_TABLES, () => [...db.tables.keys()].sort());

  const tableInfo = useTableQuery(ALL_TABLES, () => {
    const info: { name: string; rowCount: number; hash: number; columns: string[] }[] = [];
    for (const [name, table] of db.tables) {
      const rows = table.select();
      info.push({
        name,
        rowCount: rows.length,
        hash: table.getHash(),
        columns: table.schema.columns.map((c) => c.name),
      });
    }
    return info;
  });

  const selectedRows = useTableQuery(
    ALL_TABLES,
    () => {
      if (!selectedTable || !db.tables.has(selectedTable)) return [];
      const table = db.getTable(selectedTable);
      return table.select({ orderBy: table.pkCol + " DESC" });
    }
  );

  const selectedSchema = useMemo(() => {
    if (!selectedTable) return null;
    return tableInfo.find((t) => t.name === selectedTable) ?? null;
  }, [selectedTable, tableInfo]);

  return (
    <main className="tables-page">
      <h2>Synced Tables</h2>
      {tableInfo.length === 0 ? (
        <p className="tables-empty">No tables synced yet. Connect to the backend to see data.</p>
      ) : (
        <>
          <div className="tables-grid">
            {tableInfo.map((t) => (
              <button
                key={t.name}
                className={`table-card ${selectedTable === t.name ? "active" : ""}`}
                onClick={() => setSelectedTable(selectedTable === t.name ? null : t.name)}
              >
                <div className="table-card-name">{t.name}</div>
                <div className="table-card-stats">
                  <span>{t.rowCount} rows</span>
                  <span className="table-card-hash">hash: {(t.hash >>> 0).toString(16)}</span>
                </div>
                <div className="table-card-cols">
                  {t.columns.join(", ")}
                </div>
              </button>
            ))}
          </div>

          {selectedTable && selectedSchema && (
            <div className="table-detail">
              <h3>{selectedTable} ({selectedRows.length} rows)</h3>
              {selectedRows.length === 0 ? (
                <p className="tables-empty">No rows.</p>
              ) : (
                <div className="table-scroll">
                  <table className="table-data">
                    <thead>
                      <tr>
                        {selectedSchema.columns.map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selectedRows.slice(0, 200).map((row, i) => (
                        <tr key={i}>
                          {selectedSchema.columns.map((col) => (
                            <td key={col}>{formatCell(row[col])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {selectedRows.length > 200 && (
                    <p className="tables-truncated">Showing 200 of {selectedRows.length} rows</p>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </main>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(6);
  }
  const s = String(value);
  return s.length > 120 ? s.slice(0, 117) + "..." : s;
}
