export interface Column {
  name: string;
  type: "INTEGER" | "TEXT" | "REAL";
  primaryKey?: boolean;
  autoincrement?: boolean;
  notNull?: boolean;
  default?: string;
}

export interface Index {
  name: string;
  columns: string[];
}

export interface TableSchema {
  name: string;
  columns: Column[];
  indexes?: Index[];
}
