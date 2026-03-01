# smallest-llm Dashboard

Real-time training dashboard for the smallest-llm project. Built with React, Vite, TypeScript, and TradingView Lightweight Charts (WebGL-accelerated).

## Quick Start

```bash
cd dashboard
npm install
npm run dev
```

Opens at `http://localhost:5173`. Currently runs with mock data simulating a training run.

## Architecture

```
src/
├── types/                  # ALL TypeScript types live here
│   ├── metrics.ts          # Core data types: StepMetrics, LayerStat, Generation, TrainingState
│   └── chart.ts            # Chart theme config and color constants
├── hooks/
│   ├── useMetricsStore.ts  # Zustand store — single source of truth for all metrics state
│   └── useWebSocket.ts     # Data source (mock now, WebSocket later)
├── components/
│   ├── Sidebar.tsx         # Left nav: Home, Gradients pages
│   ├── Header.tsx          # Experiment name, status badge, step counter, elapsed time
│   ├── LossChart.tsx       # Train/val loss over steps (Lightweight Charts)
│   ├── LRChart.tsx         # Learning rate schedule (Lightweight Charts)
│   ├── GradientChart.tsx   # Grad norm + update/param ratio, dual axis (Lightweight Charts)
│   ├── MetricsCards.tsx    # Summary cards: loss, BPC, tokens/sec, tokens seen
│   ├── LayerStats.tsx      # Per-layer gradient/weight norms table with color bars
│   ├── GenerationPanel.tsx # Sample text generations from the model
│   └── ProgressBar.tsx     # Step progress bar with ETA
├── mock/
│   └── data.ts             # Mock data generators simulating a training run
├── App.tsx                 # Root layout: sidebar + page routing + data hook
├── App.css                 # All styles (CSS custom properties, grid layout)
└── main.tsx                # Vite entry point
```

## Key Patterns

### Types (`src/types/`)
All TypeScript interfaces and type definitions live in `src/types/`. When adding new metrics or data structures, define them here first, then use them in the store and components.

- **`metrics.ts`** — Data model types (`StepMetrics`, `LayerStat`, `Generation`, `TrainingState`, `TrainingStatus`)
- **`chart.ts`** — `CHART_THEME` (shared dark theme for all Lightweight Charts), `CHART_COLORS` (semantic color mapping)

### State Management (`useMetricsStore`)
Zustand store at `src/hooks/useMetricsStore.ts`. All metrics flow through here.

- `pushStep(metrics)` — append a new step's metrics (called ~10Hz)
- `setLayerStats(stats)` — update per-layer gradient/weight stats
- `addGeneration(gen)` — add a sample generation (keeps last 10)
- `setStatus(status)` — set training status ("training", "paused", "complete", "idle")
- `reset()` — clear all state

### Charts (Lightweight Charts)
Charts use **imperative updates** for performance — data is pushed via `series.update()` rather than React state/re-renders.

Each chart component follows this pattern:
1. `useEffect` #1: Create chart instance + series, attach ResizeObserver. Cleanup on unmount.
2. `useEffect` #2: Subscribe to Zustand store, push new data points to chart series.
3. `lastStepRef` tracks which data has already been sent to the chart to avoid duplicates.
4. Step numbers are offset by `BASE_TIME = 1_000_000` because Lightweight Charts expects ascending time values (raw step numbers 1,2,3 cause rendering issues).

### Pages
Simple state-based routing in `App.tsx` (`useState` for active page). Two pages:
- **Home** — 3x2 grid with loss, LR, metrics cards, gradient health, layer stats, generations
- **Gradients** — Dedicated 2-column view for gradient chart + layer health table

### Styling
Single CSS file (`App.css`) using CSS custom properties for theming. Dark theme. CSS Grid for layout. No CSS framework.

## How to Add Things

### Adding a new metric to an existing chart
1. Add the field to `StepMetrics` in `src/types/metrics.ts`
2. Generate mock values in `src/mock/data.ts`'s `generateStepMetrics()`
3. Store it via `pushStep` (automatic — it stores the full `StepMetrics` object)
4. In the chart component, add a new series in the first `useEffect` and push data in the second `useEffect`

### Adding a new chart component
1. Create `src/components/MyChart.tsx` following the pattern in `LossChart.tsx`
2. Use `CHART_THEME` and `CHART_COLORS` from `src/types/chart.ts`
3. Add it to the grid in `App.tsx` (add a new `grid-*` class in CSS)

### Adding a new page
1. Add the page name to the `activePage` type in `Sidebar.tsx`
2. Add a nav button in `Sidebar.tsx`
3. Create the page component in `App.tsx`
4. Add it to the page switch in `App.tsx`'s render

### Adding a new data type
1. Define the interface in `src/types/metrics.ts`
2. Add it to `TrainingState` in `src/types/metrics.ts`
3. Add a store action in `src/hooks/useMetricsStore.ts`
4. Generate mock data in `src/mock/data.ts`
5. Call the store action from `src/hooks/useWebSocket.ts`

### Connecting real WebSocket (replacing mock)
Replace the `setInterval` in `src/hooks/useWebSocket.ts` with a WebSocket connection:
```ts
const ws = new WebSocket("ws://localhost:8765");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "step") pushStep(data.metrics);
  if (data.type === "layers") setLayerStats(data.stats);
  if (data.type === "generation") addGeneration(data.generation);
};
```
The store interface stays identical — only the data source changes.

## Dependencies
- `react` / `react-dom` — UI framework
- `lightweight-charts` — WebGL-accelerated charting (TradingView)
- `zustand` — Minimal state management (~1KB)
- `vite` — Dev server + bundler
- `typescript` — Type safety
