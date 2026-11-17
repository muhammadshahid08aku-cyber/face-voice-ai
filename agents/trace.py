"""A trace records what each agent did, so every decision can be explained and audited."""
import time


class Trace(list):
    def add(self, agent: str, status: str, detail: str = "", t0: float | None = None):
        ms = round((time.perf_counter() - t0) * 1000, 1) if t0 else 0.0
        self.append({"agent": agent, "status": status, "detail": detail, "ms": ms})
