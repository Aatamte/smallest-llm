import { useRef, useState, useEffect } from "react";

export interface CompareRunsSelectorProps {
  runs: { id: number; name: string }[];
  activeRunId: number | null;
  selected: number[];
  onChange: (ids: number[]) => void;
}

const COMPARE_COLORS = ["#f97316", "#a855f7", "#14b8a6", "#e11d48", "#84cc16", "#64748b"];

export { COMPARE_COLORS };

export function CompareRunsSelector({ runs, activeRunId, selected, onChange }: CompareRunsSelectorProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handle(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [open]);

  const otherRuns = runs.filter((r) => r.id !== activeRunId);
  if (otherRuns.length === 0) return null;

  function toggle(id: number) {
    onChange(
      selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]
    );
  }

  return (
    <div ref={ref} className="compare-selector">
      <button className="compare-selector-btn" onClick={() => setOpen((v) => !v)}>
        Compare{selected.length > 0 ? ` (${selected.length})` : ""}
        <span style={{ marginLeft: 4, fontSize: 10 }}>{open ? "\u25B2" : "\u25BC"}</span>
      </button>

      {/* Selected run pills */}
      {selected.map((id, i) => {
        const run = runs.find((r) => r.id === id);
        const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
        return (
          <span key={id} className="compare-pill" style={{ borderColor: color, color }}>
            #{id} {run?.name ?? ""}
            <button className="compare-pill-x" onClick={() => toggle(id)}>&times;</button>
          </span>
        );
      })}

      {open && (
        <div className="compare-dropdown">
          {otherRuns.map((r) => (
            <label key={r.id} className="compare-dropdown-item">
              <input
                type="checkbox"
                checked={selected.includes(r.id)}
                onChange={() => toggle(r.id)}
              />
              #{r.id} {r.name}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
