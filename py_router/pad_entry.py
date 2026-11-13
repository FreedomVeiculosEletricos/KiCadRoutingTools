"""Pad entry: the joint between a track end and the pad it lands on.

KiCad (and every connectivity test here) credits a track to a pad as soon as
the track's copper touches the pad's copper. A route may therefore end with
its round cap grazing the pad edge or a corner: connected, DRC-clean, and
joined through a sliver narrower than the track. This module grades the joint
by its final geometry instead, the way Altium's router builds it:

  * the track's own cross-section, plus ``PAD_ENTRY_MARGIN_MM``, lies inside
    the pad -- the deepest point of the terminal segment's centreline is at
    least ``width/2 + margin`` from the pad outline, so the joint is as wide as
    the track and survives the fab's etch tolerance;
  * a pad narrower than that (a deliberate neck-down onto a fine pin) is met
    on its medial axis, which is the deepest the pad allows.

An entry short of that is SHALLOW. The repair (close_soft_joints) runs a
connector from the end to the pad centre -- where Altium's router starts a
pad exit and where Eagle requires a wire to end -- so the end gains a
full-width path into the pad without moving routed copper. Detection
(check_weird's ``shallow-pad-entry``), repair and the strict removal model
(which refuses to recreate one) share ``shallow_pad_entry`` so they agree.

Pads: rect, roundrect, oval and circle, any rotation. Custom-polygon pads are
not graded (no closed-form outline), as for the #416 web test.

On an SMD pad the EXIT is graded too, by Altium's SMT rules
(``grade_pad_exit``):

  * SMD Entry -- the track crosses the pad outline perpendicular to an edge,
    its centreline on the edge's straight part and its copper inside the
    edge's span (to SMD_ENTRY_OVERHANG_MM): never through a corner, never at
    an angle. The ends (short edges) are always allowed, the sides when
    SMD_ENTRY_SIDE (as Altium, a pad whose side is not > 2x its end has no
    distinct side). A round pad is met radially;
  * SMD To Corner -- the track runs straight from the outline for at least
    ``exit_distance(width)`` before its first bend, so no bend's copper sits
    on the pad edge.

pcb_modification.rebuild_pad_exits rebuilds an exit that fails either as
Altium's router lays one: from the pad centre, straight out through an
allowed edge to the exit distance, then back onto the route.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

from routing_defaults import (PAD_ENTRY_MARGIN_MM, SMD_TO_CORNER_MM, SMD_ENTRY_SIDE,
                              SMD_ENTRY_OVERHANG_MM)
from routing_utils import _to_pad_frame

#: Depth tolerance (mm): a track run exactly on a narrow pad's medial axis
#: reaches the required depth only up to float and 1 um coordinate rounding.
DEPTH_TOL = 0.002

_SHAPES = ('rect', 'roundrect', 'oval', 'circle')


def gradable_pad(pad, layer: str) -> bool:
    """A pad whose entry this module grades on ``layer``."""
    if getattr(pad, 'shape', None) not in _SHAPES:
        return False
    if getattr(pad, 'polygons', None):
        return False
    if getattr(pad, 'pad_type', None) == 'np_thru_hole':
        return False
    if not pad.size_x or not pad.size_y:
        return False
    return layer in pad.layers or any('*' in L and L.endswith('.Cu')
                                      for L in pad.layers)


def _corner_radius(pad) -> float:
    from check_drc import pad_corner_radius
    return min(pad_corner_radius(pad), pad.size_x / 2.0, pad.size_y / 2.0)


def pad_depth(pad, x: float, y: float) -> float:
    """Signed distance from (x, y) to the pad outline: positive inside (how far
    in), negative outside. Exact for a rounded rectangle at any rotation, which
    covers rect / roundrect / oval / circle."""
    lx, ly = _to_pad_frame(x, y, pad)
    rr = _corner_radius(pad)
    qx = abs(lx) - (pad.size_x / 2.0 - rr)
    qy = abs(ly) - (pad.size_y / 2.0 - rr)
    if qx <= 0 and qy <= 0:
        return rr + min(-qx, -qy)
    return rr - math.hypot(max(qx, 0.0), max(qy, 0.0))


def segment_pad_depth(pad, ax, ay, bx, by) -> float:
    """Deepest point of the segment a->b inside the pad (pad_depth's maximum).
    pad_depth is concave on a convex pad, so a ternary search is exact."""
    def f(t):
        return pad_depth(pad, ax + (bx - ax) * t, ay + (by - ay) * t)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            lo = m1
        else:
            hi = m2
    return max(f(0.0), f(1.0), f((lo + hi) / 2.0))


def required_depth(pad, width: float, margin: float = PAD_ENTRY_MARGIN_MM) -> float:
    """How deep a ``width`` track must reach into ``pad``: its half-width plus
    the margin, capped at the pad's own deepest point (its medial axis) when the
    pad is narrower than that."""
    return min(width / 2.0 + margin, min(pad.size_x, pad.size_y) / 2.0)


def shallow_pad_entry(pads: Iterable, layer: str, ex, ey, nx, ny, width,
                      margin: float = PAD_ENTRY_MARGIN_MM
                      ) -> Optional[Tuple[object, float, float]]:
    """The pad a track end (ex, ey) -- on the segment from (nx, ny) -- enters
    SHALLOW, as ``(pad, depth, required)``; None when the end touches no
    gradable pad or reaches the required depth in one it touches."""
    r = width / 2.0
    worst = None
    for pad in pads:
        if not gradable_pad(pad, layer):
            continue
        if pad_depth(pad, ex, ey) <= -r + 1e-6:
            continue  # the cap does not reach this pad's copper
        need = required_depth(pad, width, margin)
        d = segment_pad_depth(pad, nx, ny, ex, ey)
        if d >= need - DEPTH_TOL:
            return None
        if worst is None or d > worst[1]:
            worst = (pad, d, need)
    return worst


def entry_targets(pad, ex, ey, width, margin: float = PAD_ENTRY_MARGIN_MM):
    """Where a connector from a shallow end should land, best first: the pad
    centre (Altium / Eagle), then the nearest point on the way to it that meets
    the required depth (a shorter connector when the centre is blocked)."""
    cx, cy = pad.global_x, pad.global_y
    out = [(cx, cy)]
    need = required_depth(pad, width, margin)
    if pad_depth(pad, ex, ey) >= need - DEPTH_TOL:
        return out
    lo, hi = 0.0, 1.0   # depth is non-decreasing from the end to the centre
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if pad_depth(pad, ex + (cx - ex) * mid, ey + (cy - ey) * mid) >= need:
            hi = mid
        else:
            lo = mid
    px, py = ex + (cx - ex) * hi, ey + (cy - ey) * hi
    if math.hypot(px - cx, py - cy) > 1e-4:
        out.append((round(px, 4), round(py, 4)))
    return out


def on_other_track(segs: Iterable, seg, x, y, tol: float = 1e-3) -> bool:
    """Does (x, y) -- an end of ``seg`` -- lie on the centreline of another
    same-net track on its layer? Then it is a T-junction into copper that
    carries on, not an end entering a pad."""
    from geometry_utils import point_to_segment_distance
    for o in segs:
        if o is seg or o.net_id != seg.net_id or o.layer != seg.layer:
            continue
        if getattr(o, 'graphic', False):
            continue
        if point_to_segment_distance(x, y, o.start_x, o.start_y,
                                     o.end_x, o.end_y) <= tol:
            return True
    return False


def on_via(vias: Iterable, x, y) -> bool:
    """Does (x, y) lie in a via barrel (a via terminal, not a pad entry)?"""
    return any(math.hypot(x - v.x, y - v.y) <= (getattr(v, 'size', 0.0) or 0.0) / 2.0 + 0.01
               for v in vias)


# ---------------------------------------------------------------------------
# SMD exit (Altium SMD Entry / SMD To Corner)
# ---------------------------------------------------------------------------

#: Direction tolerance of a perpendicular exit (sine of 1 degree).
ANGLE_TOL = math.sin(math.radians(1.0))


def is_smd(pad) -> bool:
    """An SMD pad: no drill (Altium's SMT rules bind single-layer pads only)."""
    return not (getattr(pad, 'drill', 0) or 0) and getattr(
        pad, 'pad_type', 'smd') not in ('thru_hole', 'np_thru_hole')


def smd_gradable(pad, layer: str) -> bool:
    return is_smd(pad) and gradable_pad(pad, layer)


def exit_distance(width: float) -> float:
    """SMD To Corner: the straight run a ``width`` track keeps from the pad
    outline to its first bend -- SMD_TO_CORNER_MM, and never less than its
    half-width (a bend closer than that puts its copper on the pad edge)."""
    return max(SMD_TO_CORNER_MM, width / 2.0)


def _round_pad(pad) -> bool:
    return (getattr(pad, 'shape', None) in ('circle', 'oval')
            and abs(pad.size_x - pad.size_y) < 1e-9)


def _rot(pad):
    rr = math.radians(getattr(pad, 'rect_rotation', 0.0) or 0.0)
    return math.cos(rr), math.sin(rr)


def to_local_dir(pad, dx, dy):
    c, s = _rot(pad)
    return dx * c + dy * s, -dx * s + dy * c


def to_board_dir(pad, lx, ly):
    c, s = _rot(pad)
    return lx * c - ly * s, lx * s + ly * c


def allowed_exits(pad, allow_side: bool = SMD_ENTRY_SIDE):
    """The pad-frame unit normals a track may leave ``pad`` along, ends first.
    The ends are the short edges (all four on a square pad); the sides are
    allowed with ``allow_side`` or when they are not > 2x the ends."""
    hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
    xs = [(1.0, 0.0), (-1.0, 0.0)]          # normals of the x-edges (length 2hy)
    ys = [(0.0, 1.0), (0.0, -1.0)]          # normals of the y-edges (length 2hx)
    if abs(hx - hy) < 1e-9:
        return xs + ys
    ends, sides = (xs, ys) if hx > hy else (ys, xs)
    long_, short_ = max(hx, hy), min(hx, hy)
    if allow_side or long_ <= 2.0 * short_:
        return ends + sides
    return ends


def _collinear(ax, ay, bx, by) -> bool:
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    if la < 1e-9 or lb < 1e-9:
        return True
    return (abs(ax * by - ay * bx) <= ANGLE_TOL * la * lb
            and ax * bx + ay * by > 0)


def terminal_chain(seg, ex, ey, adj, key, vias=(), max_len: float = 8.0):
    """The track run from the free end (ex, ey) of ``seg`` outward, through
    plain degree-2 vertices of the same layer and width: ``(pts, segs,
    reason)`` with ``pts`` from the end outward. It stops at a junction, a
    via, a width change or after ``max_len`` mm (``reason``: 'junction',
    'via', 'width', 'length', 'end'). ``adj`` maps ``key(x, y)`` to the
    same-net same-layer segments with an end there."""
    pts = [(ex, ey)]
    segs = []
    s = seg
    x, y = ex, ey
    total = 0.0
    while True:
        segs.append(s)
        if key(s.start_x, s.start_y) == key(x, y):
            nx, ny = s.end_x, s.end_y
        else:
            nx, ny = s.start_x, s.start_y
        total += math.hypot(nx - x, ny - y)
        pts.append((nx, ny))
        x, y = nx, ny
        others = [o for o in adj.get(key(x, y), ()) if o is not s]
        if not others:
            return pts, segs, 'end'
        if len(others) > 1:
            return pts, segs, 'junction'
        if on_via(vias, x, y):
            return pts, segs, 'via'
        o = others[0]
        if abs(o.width - s.width) > 1e-6:
            return pts, segs, 'width'
        if total >= max_len or any(o is t for t in segs):
            return pts, segs, 'length'
        s = o


def exit_crossing(pad, pts):
    """Where a run (``pts`` from its end inside the pad outward) leaves the
    pad: ``(k, (x, y))`` with the crossing on segment pts[k]->pts[k+1]; None
    when the end is not inside the pad or the run never leaves it."""
    if pad_depth(pad, *pts[0]) <= 0:
        return None
    for k in range(len(pts) - 1):
        if pad_depth(pad, *pts[k]) > 0 and pad_depth(pad, *pts[k + 1]) <= 0:
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if pad_depth(pad, ax + (bx - ax) * mid, ay + (by - ay) * mid) > 0:
                    lo = mid
                else:
                    hi = mid
            return k, (ax + (bx - ax) * hi, ay + (by - ay) * hi)
    return None


def grade_pad_exit(pad, pts, width, allow_side: bool = SMD_ENTRY_SIDE):
    """Altium's SMD Entry and SMD To Corner on a track run entering an SMD
    ``pad``: ``pts`` from the end inside the pad outward (terminal_chain).
    Returns the set of rules it fails, of {'smd-entry', 'smd-to-corner'}; empty
    when the end is not inside the pad (a shallow entry, graded apart) or the
    run never leaves it."""
    out = set()
    cross = exit_crossing(pad, pts)
    if cross is None:
        return out
    k, (kx, ky) = cross
    (ax, ay), (bx, by) = pts[k], pts[k + 1]
    vx, vy = bx - ax, by - ay
    lvx, lvy = to_local_dir(pad, vx, vy)
    lkx, lky = _to_pad_frame(kx, ky, pad)
    lv = math.hypot(lvx, lvy)
    if _round_pad(pad):
        # radial: the run's line passes through the centre
        if abs(lkx * lvy - lky * lvx) > 0.01 * lv or lkx * lvx + lky * lvy < 0:
            out.add('smd-entry')
    else:
        hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
        rr = _corner_radius(pad)
        ix, iy, tol = hx - rr, hy - rr, 1e-3
        n = None
        if abs(lky) <= iy + tol and abs(lkx) >= ix - tol:
            n, t, span = (math.copysign(1.0, lkx), 0.0), lky, hy
        elif abs(lkx) <= ix + tol and abs(lky) >= iy - tol:
            n, t, span = (0.0, math.copysign(1.0, lky)), lkx, hx
        r = width / 2.0
        if (n is None or n not in allowed_exits(pad, allow_side)
                or abs(lvx * n[1] - lvy * n[0]) > ANGLE_TOL * lv
                or lvx * n[0] + lvy * n[1] <= 0
                or abs(t) > span - rr + tol
                or not (abs(t) + r <= span + SMD_ENTRY_OVERHANG_MM + tol
                        or (r >= span - tol
                            and abs(t) <= SMD_ENTRY_OVERHANG_MM + tol))):
            out.add('smd-entry')
    # straight run from the outline to the first bend
    run = math.hypot(bx - kx, by - ky)
    j = k + 1
    while j < len(pts) - 1:
        (px, py), (qx, qy), (sx, sy) = pts[j - 1], pts[j], pts[j + 1]
        if not _collinear(qx - px, qy - py, sx - qx, sy - qy):
            if run < exit_distance(width) - 1e-3:
                out.add('smd-to-corner')
            break
        run += math.hypot(sx - qx, sy - qy)
        j += 1
    return out


# --- exit funnels: the router lays compliant exits by construction ---------

#: Router octant of a unit grid step (rust_router router.rs octant_index); a
#: move mask's bit k allows octant k.
_OCTANT = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
           (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}



def funnel_penalty_units(config) -> int:
    """The router cost of one grid step off an exit funnel (the GridObstacleMap
    move-mask penalty): ``config.smd_exit_funnel_penalty`` vias, 0 = barred."""
    from routing_defaults import SMD_EXIT_FUNNEL_PENALTY
    vias = getattr(config, 'smd_exit_funnel_penalty', SMD_EXIT_FUNNEL_PENALTY)
    return max(0, int(round(vias * config.via_cost_units())))


def _axis_rotation(pad) -> Optional[bool]:
    """None when the pad is not at a multiple of 90 degrees; else whether its
    x and y swap on the board (90 / 270)."""
    rot = (getattr(pad, 'rect_rotation', 0.0) or 0.0) % 180.0
    if abs(rot) < 1e-6 or abs(rot - 180.0) < 1e-6:
        return False
    if abs(rot - 90.0) < 1e-6:
        return True
    return None


def exit_funnel(pad, layer_idx: int, width: float, grid_step: float,
                allow_side: bool = SMD_ENTRY_SIDE, blocked=None):
    """The move masks that make a grid route reach ``pad``'s centre only
    through an Altium-compliant exit, as ``({(gx, gy, layer_idx): mask},
    n_exits)``; ``(None, why)`` when the pad gets no funnel.

    Around the pad, the cells where a ``width`` track's copper would touch it
    are closed (mask 0), except the funnel lines: from the centre cell
    straight out along each allowed edge normal (allowed_exits) that runs on a
    grid line, to ``exit_distance(width)`` past the outline. A funnel cell
    allows only moves along its line, so a route leaves the pad perpendicular
    to an edge and keeps straight to the exit distance before it may turn
    (SMD Entry and SMD To Corner), and no route grazes, crosses or ends on
    the pad anywhere else. The centre cell allows every funnel's direction.

    A pad whose centre is off the grid gets its funnel lines on the grid
    lines nearest its axes: the line then crosses the edge a little off the
    middle, which is still straight out through the edge as long as the
    track's copper stays within the edge's span; an exit where it would not
    (a fine-pitch pad half a grid step off) is left out. Every line is graded
    as a finished exit (grade_pad_exit) before it is offered.

    ``blocked(gx, gy)``, when given, says whether the search may not enter a
    cell; an exit with a blocked cell on its line or at its mouth is left out,
    so a funnel never offers a dead end.

    Axis-aligned, non-round SMD pads only; a pad with no exit left gets no
    funnel (``why`` 'off-grid' or 'blocked')."""
    geo = _funnel_geometry(pad, width, grid_step, allow_side)
    if isinstance(geo, str):
        return None, geo
    closed, exits = geo
    lines = {}
    n_exits = 0
    n_blocked = 0
    for line, mouth, bits in exits:
        if blocked is not None and (blocked(*mouth)
                                    or any(blocked(gx, gy) for gx, gy in line[1:])):
            n_blocked += 1                # a dead end (the mouth included)
            continue
        for c in line:
            lines[c] = lines.get(c, 0) | bits
        n_exits += 1
    if not lines:
        return None, ('blocked' if n_blocked else 'off-grid')
    out = {(gx, gy, layer_idx): 0 for gx, gy in closed if (gx, gy) not in lines}
    for (gx, gy), m in lines.items():
        out[(gx, gy, layer_idx)] = m
    return out, n_exits


#: _funnel_geometry results by pad geometry, track width, grid and side rule.
#: A search installs the same pads' funnels many times (every tap, every
#: retry); the geometry never changes, only which exits are blocked.
_FUNNEL_GEOMETRY_CACHE = {}


def _funnel_geometry(pad, width: float, grid_step: float, allow_side: bool):
    """exit_funnel's board-independent half: ``(closed cells, exits)`` with
    each exit ``(line cells incl. the centre cell, mouth cell, move bits)``
    and every exit graded as laid (grade_pad_exit); or the reason the pad
    gets no funnel ('round', 'rotation', 'off-grid'). Cached."""
    key = (round(pad.global_x, 6), round(pad.global_y, 6), pad.size_x,
           pad.size_y, getattr(pad, 'rect_rotation', 0.0) or 0.0,
           getattr(pad, 'shape', None), _corner_radius(pad), round(width, 6),
           round(grid_step, 6), bool(allow_side))
    hit = _FUNNEL_GEOMETRY_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_FUNNEL_GEOMETRY_CACHE) > 50000:
        _FUNNEL_GEOMETRY_CACHE.clear()
    geo = _FUNNEL_GEOMETRY_CACHE[key] = _funnel_geometry_uncached(
        pad, width, grid_step, allow_side)
    return geo


def _funnel_geometry_uncached(pad, width, grid_step, allow_side):
    if _round_pad(pad):
        return 'round'
    swap = _axis_rotation(pad)
    if swap is None:
        return 'rotation'
    g = float(grid_step)
    cx, cy = pad.global_x, pad.global_y
    hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
    if swap:
        hx, hy = hy, hx
    inv = 1.0 / g                         # GridCoord.to_grid, to the same cell
    gcx, gcy = round(cx * inv), round(cy * inv)
    r = width / 2.0
    run = exit_distance(width)
    exits = []
    for lx, ly in allowed_exits(pad, allow_side):
        bx, by = to_board_dir(pad, lx, ly)
        nx, ny = int(round(bx)), int(round(by))
        if (nx, ny) not in _OCTANT or (nx and ny):
            continue
        half = hx if nx else hy
        bits = (1 << _OCTANT[(nx, ny)]) | (1 << _OCTANT[(-nx, -ny)])
        k = 1
        line = [(gcx, gcy)]
        while True:
            gx, gy = gcx + k * nx, gcy + k * ny
            along = (gx * g - cx) * nx + (gy * g - cy) * ny
            if along - half >= run - 1e-6:
                break                     # the funnel's mouth: free to turn
            line.append((gx, gy))
            k += 1
        # the exit as the router would lay it: the pad centre to the centre
        # cell, along the line to its mouth, then a right-angle turn
        mx, my = (gcx + k * nx) * g, (gcy + k * ny) * g
        if grade_pad_exit(pad, [(cx, cy), (gcx * g, gcy * g), (mx, my),
                                (mx - ny * g, my + nx * g)], width, allow_side):
            continue                      # off its axis by more than the edge allows
        exits.append((tuple(line), (gcx + k * nx, gcy + k * ny), bits))
    if not exits:
        return 'off-grid'
    reach = r + g / 2.0                   # a diagonal step's centreline sag
    span = int(math.ceil((max(hx, hy) + reach) / g)) + 1
    closed = tuple((gx, gy)
                   for gx in range(gcx - span, gcx + span + 1)
                   for gy in range(gcy - span, gcy + span + 1)
                   if pad_depth(pad, gx * g, gy * g) > -reach)
    return closed, tuple(exits)


def install_pad_funnel(obstacles, pad, layer_name: str, config,
                       layer_idx: int = 0) -> int:
    """Put ``pad``'s exit funnel for a ``config.track_width`` track on
    ``obstacles`` (a GridObstacleMap; its ``layer_idx`` is ``layer_name``),
    with the funnel penalty, for one search into the pad -- a plane tap's.
    Returns the number of exits, 0 when the pad gets no funnel; the caller
    clears the masks (clear_move_masks) after the search."""
    if (not smd_gradable(pad, layer_name)
            or not hasattr(obstacles, 'set_move_masks_batch')):
        return 0
    cells, n = exit_funnel(pad, layer_idx, config.track_width, config.grid_step,
                           blocked=lambda gx, gy: obstacles.is_blocked(gx, gy, layer_idx))
    if cells is None:
        return 0
    import numpy as np
    obstacles.set_move_masks_batch(np.array(
        [(gx, gy, li, m) for (gx, gy, li), m in cells.items()], dtype=np.int32))
    obstacles.set_move_mask_penalty(funnel_penalty_units(config))
    return n
