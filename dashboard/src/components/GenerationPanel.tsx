import { useCallback } from "react";
import { useAtomValue } from "jotai";
import { activeRunIdAtom } from "../storage";
import { useQuery } from "../db/hooks";
import { getGenerations } from "../db/queries";

export function GenerationPanel() {
  const runId = useAtomValue(activeRunIdAtom);
  const generations = useQuery(useCallback(() => getGenerations(runId), [runId]));

  return (
    <div className="panel">
      <h3 className="panel-title">Sample Generations</h3>
      <div className="generations-container">
        {generations.length === 0 ? (
          <div className="panel-empty">Waiting for generations...</div>
        ) : (
          generations.map((gen, i) => (
            <div key={`${gen.step}-${i}`} className="generation-item">
              <div className="generation-header">
                <span className="generation-step">step {gen.step}</span>
                <span className="generation-prompt">{gen.prompt}</span>
              </div>
              <div className="generation-text">{gen.output}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
