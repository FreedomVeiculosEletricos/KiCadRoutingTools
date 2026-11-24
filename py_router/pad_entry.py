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

An entry short of that is SHALLOW, and so is a pad no same-net track
reaches that deep at all -- the track passes its edge or bends on its corner
without ending in it (``shallow_pad_contacts``). The router does not create
one: a route ends on the pad centre (where Altium's router starts a pad exit
and Eagle requires a wire to end), a route that passes a pad of its net lays
a leg to that pad's centre, and the passes that remove copper -- the strict
removal model of the #1063 collapse and the soft-joint restore guard of the
subtractive cleanup passes -- refuse a removal that would leave one. The
detector (check_weird's ``shallow-pad-entry``) and those guards share these
predicates so they agree.

Pads: rect, roundrect, oval and circle, any rotation. Custom-polygon pads are
not graded (no closed-form outline), as for the #416 web test.

On an SMD pad the EXIT is graded too, by Altium's SMT rules
(``grade_pad_exit``):

  * SMD Entry -- the track crosses the pad outline perpendicular to an edge,
    its centreline on the edge's straight part and its copper inside the
    edge's span (to SMD_ENTRY_OVERHANG_MM). The ends (short edges) are always
    allowed, the sides when SMD_ENTRY_SIDE (as Altium, a pad whose side is
    not > 2x its end has no distinct side). A round pad is met radially. With
    corner exits (corner_exits(), the default) it may instead leave at 45
    degrees through a CORNER, its centreline on the corner's bisector to
    SMD_CORNER_EXIT_TOL_MM (corner_exit_ok); any other angle, or a 45-degree
    crossing off the corner, fails;
  * SMD To Corner -- the track runs straight from the outline for at least
    ``exit_distance(width)`` (the run's smd_to_corner(), default
    SMD_TO_CORNER_MM) before its first bend, so no bend's copper sits on the
    pad edge.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

from routing_defaults import (PAD_ENTRY_MARGIN_MM, SMD_TO_CORNER_MM, SMD_ENTRY_SIDE,
                              SMD_ENTRY_OVERHANG_MM, SMD_ENTRY_CORNER,
                              SMD_CORNER_EXIT_TOL_MM)
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


def shallow_pad_contacts(pads: Iterable, segs: Iterable, vias: Iterable = (),
                         margin: float = PAD_ENTRY_MARGIN_MM):
    """Pads joined to their net's tracks only through a shallow contact.

    shallow_pad_entry grades a track END. A track can also join a pad without
    ending in it: it passes along the pad edge, or bends on the pad corner and
    carries on, and the pad counts as connected because the copper touches.
    Here the pad is graded by the deepest copper of every same-net track on
    its layer: when some track touches the pad but none reaches its
    required_depth, the pad is SHALLOW. Copper that joins nothing outside the
    pad -- a group of tracks lying wholly inside it -- is not a joint and does
    not count. A pad with a same-net via barrel on it is joined by the via and
    not graded here.

    ``pads``, ``segs`` and ``vias`` are one net's. Returns
    ``[(pad, layer, depth, required, (x, y))]`` -- (x, y) is the deepest point
    of the deepest track."""
    segs = [s for s in segs if not getattr(s, 'graphic', False)]
    group = _track_groups(segs)
    out = []
    for pad in pads:
        layers = {s.layer for s in segs}
        for layer in sorted(layers):
            if not gradable_pad(pad, layer):
                continue
            if any(pad_depth(pad, v.x, v.y) > -(getattr(v, 'size', 0.0) or 0.0) / 2.0
                   for v in vias):
                continue
            best = None   # (depth - need, depth, need, seg)
            for s in segs:
                if s.layer != layer:
                    continue
                d = segment_pad_depth(pad, s.start_x, s.start_y, s.end_x, s.end_y)
                if d <= -s.width / 2.0 + 1e-6:
                    continue   # no copper contact
                if all(pad_depth(pad, x.start_x, x.start_y) >= 0.0
                       and pad_depth(pad, x.end_x, x.end_y) >= 0.0
                       for x in group[id(s)]):
                    continue   # joins nothing outside the pad
                need = required_depth(pad, s.width, margin)
                if best is None or d - need > best[0]:
                    best = (d - need, d, need, s)
            if best is None or best[0] >= -DEPTH_TOL:
                continue
            _, d, need, s = best
            out.append((pad, layer, d, need, _deepest_point(pad, s)))
    return out


def _track_groups(segs):
    """{id(seg): [segs of its group]}: tracks joined end to end, or an end on
    another track's body, on one layer."""
    from geometry_utils import point_to_segment_distance
    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if a.layer != b.layer:
                continue
            if any(point_to_segment_distance(x, y, c.start_x, c.start_y,
                                             c.end_x, c.end_y) <= 1e-3
                   for (x, y, c) in ((a.start_x, a.start_y, b),
                                     (a.end_x, a.end_y, b),
                                     (b.start_x, b.start_y, a),
                                     (b.end_x, b.end_y, a))):
                parent[find(i)] = find(j)
    groups = {}
    for i, s in enumerate(segs):
        groups.setdefault(find(i), []).append(s)
    return {id(s): groups[find(i)] for i, s in enumerate(segs)}


def _deepest_point(pad, s):
    """The point of segment ``s`` deepest inside ``pad`` (ternary search, as
    segment_pad_depth)."""
    ax, ay, bx, by = s.start_x, s.start_y, s.end_x, s.end_y

    def f(t):
        return pad_depth(pad, ax + (bx - ax) * t, ay + (by - ay) * t)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if f(m1) < f(m2):
            lo = m1
        else:
            hi = m2
    t = max((0.0, 1.0, (lo + hi) / 2.0), key=f)
    return (ax + (bx - ax) * t, ay + (by - ay) * t)


def entry_points(pad, ex, ey, width, margin: float = PAD_ENTRY_MARGIN_MM):
    """Where a track arriving at (ex, ey) should land on ``pad``, best first:
    the pad centre (Altium's pad exit, Eagle's connection point), then the
    first point on the way there that is ``required_depth`` deep. Empty for a
    pad this module does not grade (a custom polygon)."""
    if (getattr(pad, 'shape', None) not in _SHAPES
            or getattr(pad, 'polygons', None)
            or not getattr(pad, 'size_x', 0) or not getattr(pad, 'size_y', 0)):
        return []
    cx, cy = pad.global_x, pad.global_y
    out = [(cx, cy)]
    need = required_depth(pad, width, margin)
    lo, hi = 0.0, 1.0   # depth is non-decreasing from outside to the centre
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


def _corner_env_allows() -> bool:
    import os
    return os.environ.get('KICAD_PAD_CORNER_EXITS', '1').strip().lower() \
        not in ('0', 'false', 'off', 'no')


#: Corner exits on for the run (SMD_ENTRY_CORNER; --no-pad-corner-exits and
#: KICAD_PAD_CORNER_EXITS=0 turn them off).
_CORNER = [SMD_ENTRY_CORNER and _corner_env_allows()]


def set_corner_exits(on) -> bool:
    """Set the run's corner-exit mode; returns the previous one."""
    prev = _CORNER[0]
    _CORNER[0] = bool(on) and _corner_env_allows()
    return prev


def corner_exits() -> bool:
    return _CORNER[0]


def corner_line(pad, ex, ey):
    """The bisector of ``pad``'s corner (ex, ey) (pad-frame signs): its
    deepest point inside the pad, in the pad frame -- on the pad's medial
    axis, min(half sizes) from the outline -- and its outward pad-frame unit
    direction. A round pad's "corner" line is its 45-degree radius."""
    hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
    m = min(hx, hy)
    if _round_pad(pad):
        p0 = (0.0, 0.0)
    else:
        p0 = (ex * (hx - m), ey * (hy - m))
    r = math.sqrt(0.5)
    return p0, (ex * r, ey * r)


def corner_exit_ok(pad, kx, ky, ox, oy) -> bool:
    """Is a run crossing ``pad``'s outline at board (kx, ky), heading out
    along board (ox, oy), a corner exit: 45 degrees in the pad frame, outward
    through the corner it points at, its centreline on that corner's bisector
    (corner_line) to SMD_CORNER_EXIT_TOL_MM? False with corner exits off."""
    if not _CORNER[0]:
        return False
    lvx, lvy = to_local_dir(pad, ox, oy)
    lv = math.hypot(lvx, lvy)
    if lv < 1e-12 or abs(abs(lvx) - abs(lvy)) > 2.0 * ANGLE_TOL * lv:
        return False
    ex, ey = math.copysign(1.0, lvx), math.copysign(1.0, lvy)
    lkx, lky = _to_pad_frame(kx, ky, pad)
    if lkx * ex <= 0 or lky * ey <= 0:
        return False            # crossing off that corner's quadrant
    (px, py), (dx, dy) = corner_line(pad, ex, ey)
    return abs((lkx - px) * dy - (lky - py) * dx) <= SMD_CORNER_EXIT_TOL_MM + 1e-6


def is_smd(pad) -> bool:
    """An SMD pad: no drill (Altium's SMT rules bind single-layer pads only)."""
    return not (getattr(pad, 'drill', 0) or 0) and getattr(
        pad, 'pad_type', 'smd') not in ('thru_hole', 'np_thru_hole')


def smd_gradable(pad, layer: str) -> bool:
    return is_smd(pad) and gradable_pad(pad, layer)


#: The run's SMD To Corner distance (mm): routing_defaults.SMD_TO_CORNER_MM
#: unless the run set another (--smd-to-corner, RoutingConfig.smd_to_corner,
#: the GUI's SMD To Corner field) through set_smd_to_corner.
_SMD_TO_CORNER = SMD_TO_CORNER_MM


def set_smd_to_corner(mm) -> float:
    """Set the run's SMD To Corner distance; None or a negative value restores
    the default. Returns the previous value (pass it back to restore)."""
    global _SMD_TO_CORNER
    prev = _SMD_TO_CORNER
    _SMD_TO_CORNER = (SMD_TO_CORNER_MM if mm is None or mm < 0 else float(mm))
    return prev


def smd_to_corner() -> float:
    """The run's SMD To Corner distance (mm)."""
    return _SMD_TO_CORNER


def exit_distance(width: float, distance=None) -> float:
    """SMD To Corner: the straight run a ``width`` track keeps from the pad
    outline to its first bend -- ``distance`` (default: the run's
    smd_to_corner()), and never less than its half-width (a bend closer than
    that puts its copper on the pad edge)."""
    d = _SMD_TO_CORNER if distance is None else distance
    return max(d, width / 2.0)


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
        # radial: the run's line passes through the centre (to the side
        # overhang SMD_ENTRY_OVERHANG_MM: a grid line a quantum off an
        # off-grid pad's centre is still a straight radial exit)
        if ((abs(lkx * lvy - lky * lvx) > (SMD_ENTRY_OVERHANG_MM + 1e-3) * lv
                or lkx * lvx + lky * lvy < 0)
                and not corner_exit_ok(pad, kx, ky, vx, vy)):
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
            if not corner_exit_ok(pad, kx, ky, vx, vy):
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


def strict_exit_faults(segs, vias, pads):
    """The strict pad-exit grade (smd_exits.strict) of one net's copper:
    ``segs`` / ``vias`` / ``pads`` are that net's. Every track crossing the
    outline of a gradable pad (SMD or through-hole) is graded where it
    crosses, whatever the track's degree there:

      * ``diag-exit``   -- the crossing run is not parallel to a pad axis
        (any angle, a round pad included) and is no corner exit
        (corner_exit_ok: 45 degrees on a corner's bisector);
      * ``corner-exit`` -- it crosses off the flat part of a side: its centre
        on the corner rounding, or its copper overhanging the side by more
        than SMD_ENTRY_OVERHANG_MM (a round pad: off its centre line);
      * ``short-exit``  -- it bends, branches or meets a via less than
        exit_distance past the outline (SMD To Corner);
      * ``diag-inside`` -- a segment drawn wholly inside the pad copper that
        is not parallel to a pad axis.

    Returns a Counter of (id(pad), layer, kind). Read-only."""
    from collections import Counter, defaultdict
    out = Counter()

    def key(x, y):
        return (round(x, 4), round(y, 4))
    adj = defaultdict(list)
    for s in segs:
        adj[(s.layer, key(s.start_x, s.start_y))].append(s)
        adj[(s.layer, key(s.end_x, s.end_y))].append(s)
    via_pts = [(v.x, v.y) for v in vias or ()]
    for pad in pads or ():
        reach = math.hypot(pad.size_x, pad.size_y) / 2.0 + 0.5
        for s in segs:
            if getattr(s, 'graphic', False) or not gradable_pad(pad, s.layer):
                continue
            # coarse reject: the segment's box away from the pad
            if (max(s.start_x, s.end_x) < pad.global_x - reach
                    or min(s.start_x, s.end_x) > pad.global_x + reach
                    or max(s.start_y, s.end_y) < pad.global_y - reach
                    or min(s.start_y, s.end_y) > pad.global_y + reach):
                continue
            for kind in _strict_seg_faults(pad, s, adj, via_pts):
                out[(id(pad), s.layer, kind)] += 1
    return out


def _strict_seg_faults(pad, s, adj, via_pts):
    def key(x, y):
        return (round(x, 4), round(y, 4))
    w = s.width
    r = w / 2.0
    vx, vy = s.end_x - s.start_x, s.end_y - s.start_y
    L = math.hypot(vx, vy)
    if L < 1e-6:
        return []
    n = max(4, int(L / 0.01))
    samp = [pad_depth(pad, s.start_x + vx * i / n, s.start_y + vy * i / n)
            for i in range(n + 1)]
    if max(samp) <= -r + 1e-4:
        return []
    lvx, lvy = to_local_dir(pad, vx, vy)
    axis = abs(lvx) <= ANGLE_TOL * L or abs(lvy) <= ANGLE_TOL * L
    if samp[0] > 1e-4 and samp[-1] > 1e-4 and min(samp) > 1e-4:
        return [] if axis else ['diag-inside']
    if max(samp) <= 1e-4:
        # the copper grazes the pad, the centre line never enters it
        return [] if axis else ['diag-exit']
    faults = []
    for i in range(n):
        if (samp[i] > 0) == (samp[i + 1] > 0):
            continue
        lo, hi = i / n, (i + 1) / n
        for _ in range(30):
            m = (lo + hi) / 2
            dm = pad_depth(pad, s.start_x + vx * m, s.start_y + vy * m)
            if (dm > 0) == (samp[i] > 0):
                lo = m
            else:
                hi = m
        t = (lo + hi) / 2
        kx, ky = s.start_x + vx * t, s.start_y + vy * t
        inside_first = samp[i] > 0
        if not axis:
            ox, oy = (vx, vy) if inside_first else (-vx, -vy)
            if not corner_exit_ok(pad, kx, ky, ox, oy):
                faults.append('diag-exit')
                continue
        lkx, lky = _to_pad_frame(kx, ky, pad)
        hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
        along_x = abs(lvy) <= ANGLE_TOL * L
        lat, span = (lky, hy) if along_x else (lkx, hx)
        if not axis:
            pass                # a corner exit (corner_exit_ok above)
        elif _round_pad(pad):
            if abs(lat) > SMD_ENTRY_OVERHANG_MM + 1e-3:
                faults.append('corner-exit')
        else:
            rr = _corner_radius(pad)
            if (abs(lat) > span - rr + 1e-3
                    or abs(lat) + r > span + SMD_ENTRY_OVERHANG_MM + 1e-3):
                faults.append('corner-exit')
        # the straight run outward from the outline
        ox, oy = (s.end_x, s.end_y) if inside_first else (s.start_x, s.start_y)
        run = math.hypot(ox - kx, oy - ky)
        ux, uy = (ox - kx) / max(run, 1e-9), (oy - ky) / max(run, 1e-9)
        cur, node = s, key(ox, oy)
        need = exit_distance(w)
        guard = 0
        while run < need - 1e-3 and guard < 200:
            guard += 1
            if pad_depth(pad, *node) > 0:
                break
            if any(math.hypot(px - node[0], py - node[1]) < 1e-3 for px, py in via_pts):
                faults.append('short-exit')
                break
            nb = [q for q in adj[(s.layer, node)] if q is not cur]
            if not nb:
                break
            if len(nb) > 1:
                faults.append('short-exit')
                break
            q = nb[0]
            a, b = key(q.start_x, q.start_y), key(q.end_x, q.end_y)
            nxt = b if a == node else a
            dx, dy = nxt[0] - node[0], nxt[1] - node[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-9:
                cur, node = q, nxt
                continue
            if abs(ux * dy - uy * dx) > ANGLE_TOL * ln or ux * dx + uy * dy < 0:
                faults.append('short-exit')
                break
            run += ln
            cur, node = q, nxt
    return faults
