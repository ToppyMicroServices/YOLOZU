"""Assignment utilities (no external deps).

Implements a minimal Hungarian algorithm for min-cost matching.
Used for query-to-GT assignment in training-first scaffolds.
"""

from __future__ import annotations

from typing import List, Tuple


def _to_matrix(cost: object) -> List[List[float]]:
    if isinstance(cost, list):
        if not cost:
            return []
        return [[float(x) for x in row] for row in cost]
    # Allow torch/numpy-like objects via iteration.
    rows = []
    for row in cost:  # type: ignore[assignment]
        rows.append([float(x) for x in row])
    return rows


def linear_sum_assignment(cost_matrix: object) -> Tuple[List[int], List[int]]:
    """Solve min-cost assignment.

    Returns (row_ind, col_ind) such that each row is assigned one unique col.

    Notes:
    - Works for rectangular matrices.
    - If rows > cols, the problem is transposed internally.
    """

    a = _to_matrix(cost_matrix)
    n = len(a)
    if n == 0:
        return ([], [])
    m = len(a[0])
    if any(len(row) != m for row in a):
        raise ValueError("cost_matrix must be rectangular")

    transposed = False
    if n > m:
        # Transpose to satisfy n <= m.
        transposed = True
        a_t = [[a[i][j] for i in range(n)] for j in range(m)]
        a = a_t
        n, m = m, n

    # Hungarian algorithm (potential-based), O(n^2*m).
    # 1-indexed for classic formulation.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = matched row for column j
    row_ind: List[int] = []
    col_ind: List[int] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)

    if transposed:
        # We solved on transposed matrix: rows are original cols.
        # Convert assignment back.
        return (col_ind, row_ind)

    return (row_ind, col_ind)
