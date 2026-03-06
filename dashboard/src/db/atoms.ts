import { atom } from "jotai";

const tableVersionAtoms: Record<string, ReturnType<typeof atom<number>>> = {};

export function getTableVersionAtom(name: string) {
  if (!tableVersionAtoms[name]) {
    tableVersionAtoms[name] = atom(0);
  }
  return tableVersionAtoms[name];
}
