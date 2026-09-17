"""
Shared path-following geometry used by both classical controllers
(AdaptivePurePursuitController) and the RL environment (PathTrackingEnv).

Kept in one place so cross-track-error / heading-error / curvature are
computed identically everywhere -- required for RL-vs-classical benchmarks
in run_evaluate_RL.py to be apples-to-apples.
"""

import math
from typing import Tuple

import numpy as np


def closest_segment_index(
    pts: np.ndarray, pos: Tuple[float, float], search_start: int = 0, search_ahead: int = 15
) -> int:
    """
    Find the path segment closest to `pos`, searching a local window around
    `search_start` (not the whole path) so a self-intersecting path doesn't
    cause the tracked index to jump backward to an earlier crossing.
    """
    window_start = max(0, search_start - 2)
    window_end = min(len(pts), search_start + search_ahead)

    best_idx = search_start
    best_dist = float("inf")
    for i in range(window_start, max(window_end - 1, window_start + 1)):
        dist = point_to_segment_distance(np.asarray(pos), pts[i], pts[min(i + 1, len(pts) - 1)])
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def path_tracking_error(
    pts: np.ndarray, pos: Tuple[float, float], theta: float, segment_idx: int
) -> Tuple[float, float, float, float]:
    """
    Compute signed cross-track error, heading error, and the projected
    point's arc-length offset within its segment, relative to the segment
    starting at `segment_idx`.

    Returns:
        cte: signed cross-track error (+ = left of path, - = right)
        heading_err: path heading - vehicle heading, wrapped to [-pi, pi]
        t: fractional position of the projection within the segment [0, 1]
        seg_len: length of the segment used
    """
    pos = np.asarray(pos, dtype=float)
    i = min(segment_idx, len(pts) - 2)
    a, b = pts[i], pts[min(i + 1, len(pts) - 1)]

    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:
        t = 0.0
        proj = a
    else:
        t = float(np.clip(np.dot(pos - a, seg) / (seg_len ** 2), 0.0, 1.0))
        proj = a + t * seg

    path_heading = math.atan2(seg[1], seg[0])
    normal = np.array([-math.sin(path_heading), math.cos(path_heading)])
    cte = float(np.dot(pos - proj, normal))

    heading_err = path_heading - theta
    heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

    return cte, heading_err, t, seg_len


def estimate_curvature(idx: int, pts: np.ndarray) -> float:
    """
    Approximate path curvature ahead of `idx` using the turning angle
    between the two segments idx->idx+1 and idx+1->idx+2, normalized by
    the total length of those two segments.
    """
    n = len(pts)
    i0, i1, i2 = idx, min(idx + 1, n - 1), min(idx + 2, n - 1)
    if i0 == i1 or i1 == i2:
        return 0.0

    ab = pts[i1] - pts[i0]
    bc = pts[i2] - pts[i1]

    angle1 = math.atan2(ab[1], ab[0])
    angle2 = math.atan2(bc[1], bc[0])
    angle_diff = abs(angle2 - angle1)
    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

    total_dist = float(np.linalg.norm(ab) + np.linalg.norm(bc))
    if total_dist < 1e-3:
        return 0.0
    return angle_diff / total_dist
