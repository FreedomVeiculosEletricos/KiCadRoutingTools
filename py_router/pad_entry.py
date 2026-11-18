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
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

from routing_defaults import PAD_ENTRY_MARGIN_MM
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
