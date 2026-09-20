"""
Kinematic (curvature-aware) path smoothing.

AStarPlanner._smooth_path() does greedy line-of-sight string-pulling: it
picks the FEWEST waypoints connectable by straight, collision-free lines.
That minimizes path length, but is completely blind to the vehicle's
turning radius -- at a tight pinch point (e.g. a narrow corridor mouth) it
can produce a near-right-angle vertex that no bicycle-model vehicle can
actually follow at any reasonable speed. Trackers (PID/PurePursuit/RL)
given that path have no choice but to cut the corner, which is exactly
what causes collisions at sharp turns close to walls.

This module rounds sharp vertices with circular arcs sized to the
vehicle's true minimum turning radius (wheelbase / tan(max_steering_angle)),
so the path is kinematically followable. Where the available space is too
tight for a full-radius arc (can genuinely happen in a narrow passage),
the radius is shrunk via bisection until a collision-free arc is found, or
the original sharp vertex is kept unmodified as a fallback -- smoothing
never turns a valid (if awkward) path into an invalid one.
"""

import math
from typing import List, Tuple

import numpy as np

from src.core.map import Map2D
from src.planning.base_planner import Path, PathPoint


def smooth_path_kinematic(
    path: Path,
    map_env: Map2D,
    min_turn_radius: float,
    min_turn_angle_deg: float = 5.0,
    samples_per_meter: float = 1.0,
    min_radius_floor: float = 0.3,
    radius_shrink_factor: float = 0.7,
) -> Path:
    """
    Round every interior vertex of `path` sharper than min_turn_angle_deg
    with a circular arc of radius >= min_radius_floor, preferring
    min_turn_radius.

    Args:
        path: waypoints AFTER line-of-sight string-pulling (i.e. the
              reduced, sharp-cornered polyline -- this is meant to run
              before arc-length resampling, not after).
        map_env: used to collision-check candidate arcs (default safety
              margin, same as the rest of the planned path).
        min_turn_radius: vehicle's minimum turning radius, in meters.
        min_turn_angle_deg: vertices with a smaller turn than this are
              left as straight-line joins (not worth curving).
        samples_per_meter: arc point density for collision checking / the
              resulting polyline.
        min_radius_floor: give up shrinking below this radius and fall
              back to the original sharp vertex.
        radius_shrink_factor: multiplicative shrink step when a candidate
              arc collides with an obstacle.
    """
    pts = path.to_array()
    n = len(pts)
    if n < 3:
        return path

    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    new_points: List[PathPoint] = [PathPoint(float(pts[0][0]), float(pts[0][1]))]

    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        dir_in = b - a
        dir_out = c - b
        len_in = float(np.linalg.norm(dir_in))
        len_out = float(np.linalg.norm(dir_out))
        if len_in < 1e-6 or len_out < 1e-6:
            new_points.append(PathPoint(float(b[0]), float(b[1])))
            continue

        dir_in_n = dir_in / len_in
        dir_out_n = dir_out / len_out
        delta = _signed_angle(dir_in_n, dir_out_n)

        if abs(math.degrees(delta)) < min_turn_angle_deg:
            new_points.append(PathPoint(float(b[0]), float(b[1])))
            continue

        # Each vertex may only consume up to 40% of each adjacent segment
        # for its own fillet, so two consecutive sharp vertices sharing a
        # segment can never overlap (0.4 + 0.4 < 1.0).
        avail_in = 0.4 * seg_lens[i - 1]
        avail_out = 0.4 * seg_lens[i]

        arc_pts = None
        radius = min_turn_radius
        while radius >= min_radius_floor:
            t = radius * math.tan(abs(delta) / 2.0)
            if t <= avail_in and t <= avail_out:
                candidate = _build_arc(b, dir_in_n, dir_out_n, delta, radius, t, samples_per_meter)
                if _arc_collision_free(candidate, map_env):
                    arc_pts = candidate
                    break
            radius *= radius_shrink_factor

        if arc_pts is None:
            # No collision-free arc fits here even at the radius floor --
            # keep the original (already collision-free) sharp vertex.
            # The path is still valid; it just needs a slow, careful turn.
            new_points.append(PathPoint(float(b[0]), float(b[1])))
        else:
            for (x, y) in arc_pts:
                new_points.append(PathPoint(float(x), float(y)))

    new_points.append(PathPoint(float(pts[-1][0]), float(pts[-1][1])))
    return Path(new_points)


def _signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Signed angle (radians, in (-pi, pi]) to rotate v1 onto v2."""
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    d = a2 - a1
    return math.atan2(math.sin(d), math.cos(d))


def _rotate90(v: np.ndarray, ccw: bool) -> np.ndarray:
    x, y = v
    return np.array([-y, x]) if ccw else np.array([y, -x])


def _build_arc(
    vertex: np.ndarray,
    dir_in_n: np.ndarray,
    dir_out_n: np.ndarray,
    delta: float,
    radius: float,
    tangent_len: float,
    samples_per_meter: float,
) -> List[Tuple[float, float]]:
    """Circular arc tangent to the incoming/outgoing directions at the
    vertex, turning through signed angle `delta`. Standard corner-fillet
    construction: tangent points sit `tangent_len` back/forward along each
    original segment, and rotating the incoming tangent point around the
    arc center by `delta` lands exactly on the outgoing tangent point."""
    ccw = delta > 0
    A = vertex - dir_in_n * tangent_len
    B = vertex + dir_out_n * tangent_len
    n_in = _rotate90(dir_in_n, ccw)
    center = A + n_in * radius

    angle_a = math.atan2(A[1] - center[1], A[0] - center[0])
    arc_len = abs(delta) * radius
    n_samples = max(2, int(arc_len * samples_per_meter) + 1)

    pts = []
    for k in range(n_samples):
        frac = k / (n_samples - 1)
        ang = angle_a + frac * delta
        pts.append((center[0] + radius * math.cos(ang), center[1] + radius * math.sin(ang)))
    # Force exact endpoints (avoids float drift vs. the straight segments
    # this arc is spliced between).
    pts[0] = (float(A[0]), float(A[1]))
    pts[-1] = (float(B[0]), float(B[1]))
    return pts


def _arc_collision_free(arc_pts: List[Tuple[float, float]], map_env: Map2D) -> bool:
    for (x, y) in arc_pts:
        if map_env.is_collision(x, y):
            return False
    return True


def smooth_entry_heading(
    path: Path,
    map_env: Map2D,
    min_turn_radius: float,
    current_heading: float,
    min_turn_angle_deg: float = 5.0,
    samples_per_meter: float = 1.0,
    min_radius_fraction: float = 0.5,
    radius_shrink_factor: float = 0.8,
    reconnect_search_waypoints: int = 15,
) -> Path:
    """
    Round the transition between the vehicle's CURRENT heading/position and
    the start of a freshly (re)planned path.

    smooth_path_kinematic() only rounds INTERIOR vertices -- it has no
    notion of the vehicle's actual momentum, so a path whose very first
    segment points in a drastically different direction from where the
    vehicle is already heading (a common outcome of online fog-of-war
    replanning: A* just finds a new geometric route from the vehicle's
    position to the goal, with zero regard for which way the vehicle
    happens to be pointed right now) still forces an instant direction
    change. If that happens to occur right next to a wall -- which it
    often does, since a replan is usually triggered by a wall appearing in
    the way -- there is no room to execute it and a collision follows.
    This is the near-180-degree "sudden U-turn" failure mode.

    Builds a circular arc starting exactly at the vehicle's current
    position, tangent to current_heading, curving until its own tangent
    direction matches the path's initial direction, then reconnects to
    the path at whichever of the next few waypoints has a collision-free
    straight bridge back to it (not necessarily the closest one, and not
    necessarily the very first waypoint -- a large heading mismatch can
    need more than one short resampled segment's worth of turning room).

    For a turn near +-180 degrees, which side has room to swing is NOT
    reliably indicated by the sign of the shortest-rotation angle -- e.g.
    delta=+179 deg and delta=-179 deg are almost the same physical turn,
    but curve to opposite sides, and only one of those sides might be
    open space (the other could be a wall right next to the vehicle,
    which is exactly the situation a replan is usually triggered by).
    So both the natural (shortest) rotation direction AND its mirror
    (delta -/+ 360 deg, i.e. sweeping the long way around) are tried at
    every candidate radius, and the best (largest-radius) collision-free
    result across both is kept. Also refuses to accept an arc tighter
    than min_radius_fraction * min_turn_radius: a "successful" fillet at
    a fraction of the vehicle's real minimum radius isn't actually
    drivable, it just LOOKS like a fix. Falls back to the original path,
    unmodified, if no such arc exists in either direction.
    """
    pts = path.to_array()
    n = len(pts)
    if n < 2:
        return path

    A = pts[0]
    dir_out = pts[1] - A
    len_out = float(np.linalg.norm(dir_out))
    if len_out < 1e-6:
        return path
    dir_out_n = dir_out / len_out

    u = np.array([math.cos(current_heading), math.sin(current_heading)])
    delta = _signed_angle(u, dir_out_n)
    if abs(math.degrees(delta)) < min_turn_angle_deg:
        return path  # already roughly aligned with current heading

    delta_mirror = delta - math.copysign(2 * math.pi, delta)
    search_limit = min(reconnect_search_waypoints, n - 1)
    min_radius = min_turn_radius * min_radius_fraction

    radius = min_turn_radius
    while radius >= min_radius:
        for cand_delta in (delta, delta_mirror):
            arc_pts = _build_entry_arc(A, current_heading, cand_delta, radius, samples_per_meter)
            if not _arc_collision_free(arc_pts, map_env):
                continue
            end = arc_pts[-1]
            # The arc's own collision-freeness says nothing about the
            # straight bridge segment from its endpoint back onto the
            # original path -- that bridge can still cut through a wall
            # in a tight corridor, so it must be checked too.
            for idx in _reconnect_candidates_by_distance(end, pts, search_limit):
                if map_env.is_path_collision_free(end[0], end[1], pts[idx][0], pts[idx][1]):
                    candidate_points = (
                        [PathPoint(float(x), float(y)) for x, y in arc_pts]
                        + list(path.points[idx:])
                    )
                    return Path(candidate_points)
        radius *= radius_shrink_factor

    return path


def _build_entry_arc(
    start: np.ndarray,
    heading: float,
    delta: float,
    radius: float,
    samples_per_meter: float,
) -> List[Tuple[float, float]]:
    """Arc starting exactly at `start`, tangent to `heading`, sweeping
    through signed angle `delta` (no backward offset -- unlike an interior
    corner fillet, there's nothing before the vehicle's current position
    to anchor a symmetric fillet to)."""
    u = np.array([math.cos(heading), math.sin(heading)])
    ccw = delta > 0
    n_u = _rotate90(u, ccw)
    center = start + n_u * radius

    angle_a = math.atan2(start[1] - center[1], start[0] - center[0])
    arc_len = abs(delta) * radius
    n_samples = max(2, int(arc_len * samples_per_meter) + 1)

    pts = []
    for k in range(n_samples):
        frac = k / (n_samples - 1)
        ang = angle_a + frac * delta
        pts.append((center[0] + radius * math.cos(ang), center[1] + radius * math.sin(ang)))
    pts[0] = (float(start[0]), float(start[1]))
    return pts


def _reconnect_candidates_by_distance(
    point: Tuple[float, float], pts: np.ndarray, search_limit: int
) -> List[int]:
    """Indices (1..search_limit) of original-path waypoints, nearest to
    `point` first -- candidates for where the entry arc splices back into
    the path. Caller tries them in order since the nearest one isn't
    always reachable by a collision-free straight bridge."""
    p = np.array(point)
    dists = [(float(np.linalg.norm(pts[i] - p)), i) for i in range(1, search_limit + 1)]
    dists.sort(key=lambda pair: pair[0])
    return [i for _, i in dists]
