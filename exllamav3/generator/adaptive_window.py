from __future__ import annotations
import os

"""
Throughput-optimal verification window for block drafters (DFlash): the drafter always produces its full
block, but verifying fewer positions makes the target's matmuls cheaper (they are decode-bound beyond two
rows). The per-position acceptance probability is tracked online and each round verifies the window that
maximizes expected tokens per unit of round time. Verification stays exact; only the window changes.
"""

# Relative round time by verified rows (target forward rows = window + 1), from the RDNA3 EXL3 matmul row
# scaling (4 / 5 / 6 / 8 rows = 21.6 / 22.9 / 24.1 / 26.6 ms per pass) plus the fixed part of a round
# (draft forward, attention, norms, dispatch; ~9.3 ms at 36 ms per round). Only the ratios matter.
# Windows whose rows land in the same kernel instance as a larger one (7 rows run as 8) are not offered.
_DEFAULT_COST = {4: 30.9, 5: 32.2, 6: 33.4, 8: 35.9}


def _parse_cost(s: str) -> dict:
    out = {}
    for item in s.split(","):
        r, t = item.split(":")
        out[int(r)] = float(t)
    return out


class AdaptiveWindow:

    def __init__(self, max_window: int, alpha: float = 0.05, prior: float = 0.6, explore: int = 6):
        self.max_window = max_window
        self.alpha = alpha
        # Every explore-th round verifies the full block: positions past a shortened window are otherwise
        # never observed and their estimates would go stale (the window could not grow back)
        self.explore = int(os.environ.get("EXL3_ADAPT_EXPLORE", explore))
        self.rounds = 0
        # q[i]: P(draft position i + 1 accepted | positions 1..i accepted), EMA over rounds that observed it
        self.q = [prior] * max_window
        cost = os.environ.get("EXL3_ADAPT_COST")
        cost = _parse_cost(cost) if cost else _DEFAULT_COST
        self.options = sorted((rows - 1, t) for rows, t in cost.items() if 1 <= rows - 1 <= max_window)
        if not any(w == max_window for w, _ in self.options):
            self.options.append((max_window, max(t for _, t in self.options)))

    def choose(self) -> int:
        self.rounds += 1
        if self.explore > 0 and self.rounds % self.explore == 0:
            return self.max_window
        best_w, best_rate = self.max_window, -1.0
        for w, t in self.options:
            p, e = 1.0, 1.0
            for i in range(w):
                p *= self.q[i]
                e += p
            rate = e / t
            if rate > best_rate:
                best_w, best_rate = w, rate
        return best_w

    def update(self, window: int, accepted: int):
        # Positions 1..accepted were accepted; position accepted + 1 (if verified) was rejected; later
        # positions were not observed
        a = self.alpha
        for i in range(min(accepted, window)):
            self.q[i] += a * (1.0 - self.q[i])
        if accepted < window:
            self.q[accepted] += a * (0.0 - self.q[accepted])
