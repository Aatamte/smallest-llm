import { useEffect, useRef } from "react";

interface WeightHeatmapProps {
  name: string;
  shape: number[];
  data: number[][];
}

/** Blue (negative) → black (zero) → red (positive) diverging colormap. */
function valueToRGB(value: number, min: number, max: number): [number, number, number] {
  const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
  const t = value / absMax; // -1 to 1

  if (t >= 0) {
    // black → red
    const r = Math.round(t * 255);
    return [r, 0, 0];
  } else {
    // black → blue
    const b = Math.round(-t * 255);
    return [0, 0, b];
  }
}

export function WeightHeatmap({ name, shape, data }: WeightHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;

    const cellSize = Math.max(2, Math.min(6, Math.floor(256 / Math.max(rows, cols))));
    const width = cols * cellSize;
    const height = rows * cellSize;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Find min/max
    let min = Infinity;
    let max = -Infinity;
    for (const row of data) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [red, green, blue] = valueToRGB(data[r][c], min, max);
        // Fill the cell
        for (let dy = 0; dy < cellSize; dy++) {
          for (let dx = 0; dx < cellSize; dx++) {
            const px = (r * cellSize + dy) * width + (c * cellSize + dx);
            pixels[px * 4] = red;
            pixels[px * 4 + 1] = green;
            pixels[px * 4 + 2] = blue;
            pixels[px * 4 + 3] = 255;
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data, rows, cols]);

  const shapeStr = shape.join(" x ");
  const shortName = name.replace(/^blocks\.(\d+)\./, "[$1] ");

  return (
    <div className="weight-heatmap">
      <div className="weight-heatmap-label">{shortName}</div>
      <canvas ref={canvasRef} style={{ display: "block", imageRendering: "pixelated" }} />
      <div className="weight-heatmap-shape">{shapeStr}</div>
    </div>
  );
}
