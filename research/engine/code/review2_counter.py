"""Fixed-clock bounded-counter compatibility; all arithmetic is float64.

This is a specialization of bounded-error shape-constrained feasibility.
The offset is free and all readings, including endpoints, may move by epsilon.
"""
import numpy as np
from scipy.optimize import linprog


def repair_radius(t, y, cap):
    """Minimum uniform repair and a violating pair in O(n) work / O(1) state."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) != len(y) or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('At least two finite readings at strictly increasing times required')
    if not np.isfinite(t).all() or not np.isfinite(y).all() or not np.isfinite(cap) or cap < 0:
        raise ValueError('Finite data and a nonnegative cap required')
    q = y - cap * (t - t[0]) / 3600
    min_q = max_y = 0
    worst, pair, kind = 0., (0, 0), 'none'
    for j in range(1, len(t)):
        for gap, i, label in [(y[max_y] - y[j], max_y, 'decrease'),
                               (q[j] - q[min_q], min_q, 'cap')]:
            if gap > worst:
                worst, pair, kind = float(gap), (int(i), j), label
        if y[j] > y[max_y]:
            max_y = j
        if q[j] < q[min_q]:
            min_q = j
    return worst / 2, pair, kind


def adjacent_radius(t, y, cap):
    dt, dy = np.diff(t), np.diff(y)
    return float(max(0., np.max(-dy), np.max(dy - cap * dt / 3600)) / 2)


def feasible_witness(t, y, cap, eps):
    """Lower feasible envelope, only intended for small validation cases."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    return np.max(y[None, :] - eps - cap * np.maximum(t[None, :] - t[:, None], 0) / 3600, axis=1)


def lp_radius(t, y, cap):
    """Independent LP: z_i free, eps >= 0, adjacent bounded increments."""
    n = len(y)
    rows, rhs = [], []
    for i in range(n):
        row = np.zeros(n + 1); row[i] = 1; row[-1] = -1
        rows.append(row); rhs.append(y[i])
        row = np.zeros(n + 1); row[i] = -1; row[-1] = -1
        rows.append(row); rhs.append(-y[i])
    for i in range(n - 1):
        row = np.zeros(n + 1); row[i + 1] = 1; row[i] = -1
        rows.append(row); rhs.append(cap * (t[i + 1] - t[i]) / 3600)
        rows.append(-row); rhs.append(0.)
    objective = np.zeros(n + 1); objective[-1] = 1
    result = linprog(objective, A_ub=rows, b_ub=rhs,
                     bounds=[(None, None)] * n + [(0, None)], method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)
