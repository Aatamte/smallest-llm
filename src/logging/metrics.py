"""Metric tracking utilities."""

from collections import defaultdict, deque


class RunningMean:
    """Efficient windowed mean computation."""

    def __init__(self, window_size: int = 100):
        self.window: deque = deque(maxlen=window_size)

    def update(self, value: float):
        self.window.append(value)

    @property
    def mean(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0

    @property
    def last(self) -> float:
        return self.window[-1] if self.window else 0.0

    def reset(self):
        self.window.clear()


class MetricTracker:
    """Track multiple named metrics with smoothing."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: dict[str, RunningMean] = defaultdict(
            lambda: RunningMean(self.window_size)
        )

    def update(self, name: str, value: float):
        self._metrics[name].update(value)

    def update_many(self, metrics: dict[str, float]):
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.update(name, value)

    def get_mean(self, name: str) -> float:
        return self._metrics[name].mean

    def get_last(self, name: str) -> float:
        return self._metrics[name].last

    def get_all_means(self) -> dict[str, float]:
        return {name: rm.mean for name, rm in self._metrics.items()}

    def reset(self):
        self._metrics.clear()
