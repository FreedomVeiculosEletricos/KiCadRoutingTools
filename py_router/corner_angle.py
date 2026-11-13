"""Corner angles: the interior angle where same-net tracks meet.

Two same-net tracks that meet at an interior angle under 90 degrees leave an
acute notch in the copper outline:

  * the notch traps etchant (an acid trap) and over-etches, or under-etches
    and leaves a sliver; heavy copper etches longer and suffers more;
  * the copper point outside it is thin, holds poorly to the laminate, and
    the solder mask thins over its edges;
  * current crowds at the inner corner, and a sharp point concentrates the
    electric field.

Commercial tools grade it: Altium's Acute Angle manufacturing rule (90
degrees by default) and the Recorner / gloss passes that remove such
corners. Here GridRouter keeps the router from laying one
(``max_turn_units`` / ``acute_turn_cost``), ``acute_corners`` finds what is
on a board (check_weird's ``acute-corner``), and
``pcb_modification.fix_acute_corners`` cuts what it can.

A corner is graded where track ENDS meet (``corner``) and where a track end
lands on another same-net track's body (``tee``). A corner inside the net's
own pad or via copper is not graded: the pad or the via barrel fills the
notch.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

#: Coordinate key precision (decimals of a mm) for track ends that meet: 1 um,
#: so ends a rewriting pass left a rounding apart (0.2 um) are one joint.
_KEY_DECIMALS = 3
#: A track end within this distance (mm) of another track's centreline lies
#: on its body (a T joint).
TEE_TOL = 1e-3
#: Angles are graded to this tolerance (degrees), so an exact 90-degree corner
#: on a rounded coordinate passes a 90-degree rule.
ANGLE_TOL_DEG = 0.5


class AcuteCorner(NamedTuple):
    layer: str
    x: float
    y: float
    angle: float          # interior angle, degrees
    kind: str             # 'corner' (ends meet) or 'tee' (end on a body)
    segs: Tuple           # the two tracks that form it


def _key(x: float, y: float) -> Tuple[float, float]:
    return (round(x, _KEY_DECIMALS), round(y, _KEY_DECIMALS))


def interior_angle(ux: float, uy: float, vx: float, vy: float) -> float:
    """Angle (degrees, 0..180) between two directions leaving one point."""
    lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
    if lu < 1e-12 or lv < 1e-12:
        return 180.0
    c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
    return math.degrees(math.acos(c))


def _away(seg, x: float, y: float) -> Tuple[float, float]:
    """Direction of ``seg`` leaving its end (x, y)."""
    if _key(seg.start_x, seg.start_y) == _key(x, y):
        return seg.end_x - seg.start_x, seg.end_y - seg.start_y
    return seg.start_x - seg.end_x, seg.start_y - seg.end_y


def _covered(x: float, y: float, layer: str, vias: Sequence, pads: Sequence) -> bool:
    """Does the net's own pad or via copper hold (x, y) on ``layer``?"""
    for v in vias:
        if math.hypot(x - v.x, y - v.y) <= (getattr(v, 'size', 0.0) or 0.0) / 2.0 + 1e-3:
            return True
    if pads:
        from connectivity import endpoint_reaches_pad
        for p in pads:
            try:
                if endpoint_reaches_pad(x, y, 1e-3, [layer], p):
                    return True
            except Exception:
                continue
    return False


def _pt_seg(px, py, ax, ay, bx, by) -> Tuple[float, float]:
    """(distance, parameter t) of point P to segment AB."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-18:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def acute_corners(segs: Iterable, vias: Sequence = (), pads: Sequence = (),
                  min_angle: float = 90.0) -> List[AcuteCorner]:
    """Every corner of one net's tracks whose interior angle is under
    ``min_angle`` degrees. ``segs`` / ``vias`` / ``pads`` are one net's;
    graphic copper is ignored. At a joint of three or more ends, each pair of
    neighbouring tracks around it is graded."""
    if not min_angle or min_angle <= 0:
        return []
    limit = float(min_angle) - ANGLE_TOL_DEG
    by_layer = defaultdict(list)
    for s in segs:
        if getattr(s, 'graphic', False):
            continue
        if math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) < 1e-6:
            continue
        by_layer[s.layer].append(s)
    out: List[AcuteCorner] = []
    for layer, lsegs in by_layer.items():
        ends = defaultdict(list)
        for s in lsegs:
            ends[_key(s.start_x, s.start_y)].append(s)
            ends[_key(s.end_x, s.end_y)].append(s)
        for (x, y), here in ends.items():
            if len(here) < 2 or _covered(x, y, layer, vias, pads):
                continue
            dirs = []
            for s in here:
                dx, dy = _away(s, x, y)
                dirs.append((math.atan2(dy, dx), s, dx, dy))
            dirs.sort(key=lambda d: d[0])
            pairs = ([(dirs[0], dirs[1])] if len(dirs) == 2 else
                     [(dirs[i], dirs[(i + 1) % len(dirs)]) for i in range(len(dirs))])
            for a, b in pairs:
                ang = interior_angle(a[2], a[3], b[2], b[3])
                if ang < limit:
                    out.append(AcuteCorner(layer, x, y, ang, 'corner', (a[1], b[1])))
        # T joints: an end on another track's body (bucketed by 1 mm cell)
        cell = 1.0
        buckets = defaultdict(list)
        for h in lsegs:
            for gx in range(int(math.floor(min(h.start_x, h.end_x) / cell)),
                            int(math.floor(max(h.start_x, h.end_x) / cell)) + 1):
                for gy in range(int(math.floor(min(h.start_y, h.end_y) / cell)),
                                int(math.floor(max(h.start_y, h.end_y) / cell)) + 1):
                    buckets[(gx, gy)].append(h)
        for s in lsegs:
            for ex, ey in ((s.start_x, s.start_y), (s.end_x, s.end_y)):
                for h in buckets.get((int(math.floor(ex / cell)),
                                      int(math.floor(ey / cell))), ()):
                    if h is s:
                        continue
                    if (max(h.start_x, h.end_x) < ex - TEE_TOL or min(h.start_x, h.end_x) > ex + TEE_TOL
                            or max(h.start_y, h.end_y) < ey - TEE_TOL
                            or min(h.start_y, h.end_y) > ey + TEE_TOL):
                        continue
                    d, t = _pt_seg(ex, ey, h.start_x, h.start_y, h.end_x, h.end_y)
                    if d > TEE_TOL or _key(ex, ey) in (_key(h.start_x, h.start_y),
                                                      _key(h.end_x, h.end_y)):
                        continue
                    if _covered(ex, ey, layer, vias, pads):
                        continue
                    sx, sy = _away(s, ex, ey)
                    hx, hy = h.end_x - h.start_x, h.end_y - h.start_y
                    ang = interior_angle(sx, sy, hx, hy)
                    ang = min(ang, 180.0 - ang)
                    # collinear: copper lying over copper -- a fold reported
                    # at its vertex, or check_weird's stacked-copper
                    if ang < 1.0:
                        continue
                    if ang < limit:
                        out.append(AcuteCorner(layer, ex, ey, ang, 'tee', (s, h)))
    return out


def corner_cut_candidates(V, A, C, min_angle: float, step: float,
                          extra_dirs: Optional[Sequence[Tuple[float, float]]] = None,
                          width: float = 0.0):
    """Cuts that replace the corner A-V-C (interior angle under ``min_angle``)
    by A-P-Q-C, P on VA and Q on VC, with both new corners at ``min_angle``
    or wider. The cut P-Q runs along one of the 8 octolinear directions (and
    ``extra_dirs``); P walks VA from A towards V in ``step`` increments.
    Yields (P, Q), the shortest resulting path first -- the largest cut, the
    gloss Altium applies. P may be A and Q may be C (the corner and its legs
    are replaced by one straight track).

    A track has ``width``: its copper's inner edge turns where the legs'
    inner edges meet, (width/2) / sin(angle/2) from V. A cut nearer V than
    that only chamfers the outside and leaves the same acute notch, so the
    cut's centreline must lie at least (width/2) * (1/sin(angle/2) - 1) from
    V."""
    vx, vy = V
    ax, ay = A[0] - vx, A[1] - vy
    cx, cy = C[0] - vx, C[1] - vy
    la, lc = math.hypot(ax, ay), math.hypot(cx, cy)
    if la < 1e-9 or lc < 1e-9:
        return []
    au = (ax / la, ay / la)
    cu = (cx / lc, cy / lc)
    dirs = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]
    for d in extra_dirs or ():
        ld = math.hypot(*d)
        if ld > 1e-12:
            dirs.append((d[0] / ld, d[1] / ld))
    limit = float(min_angle) - ANGLE_TOL_DEG
    half = math.radians(interior_angle(au[0], au[1], cu[0], cu[1])) / 2.0
    reach = (width / 2.0) * (1.0 / max(math.sin(half), 1e-6) - 1.0) if width > 0 else 0.0
    n = max(1, int(la / max(step, 1e-3)))
    cands = []
    for i in range(n, 0, -1):
        d1 = la * i / n
        px, py = vx + au[0] * d1, vy + au[1] * d1
        for ux, uy in dirs:
            # P + t*u = V + s*cu  ->  solve for t, s
            det = ux * (-cu[1]) - uy * (-cu[0])
            if abs(det) < 1e-12:
                continue
            rx, ry = vx - px, vy - py
            t = (rx * (-cu[1]) - ry * (-cu[0])) / det
            s = (ux * ry - uy * rx) / det
            if t <= 1e-6 or s <= 1e-6 or s > lc + 1e-6:
                continue
            qx, qy = vx + cu[0] * s, vy + cu[1] * s
            # distance from V to the cut's line
            if abs((vx - px) * uy - (vy - py) * ux) < reach - 1e-6:
                continue
            # P's corner: towards A (away from V) and towards Q; Q's: towards
            # P and towards C. At A or C themselves the caller grades the
            # joint with the tracks beyond.
            if d1 < la - 1e-6 and interior_angle(au[0], au[1], ux, uy) < limit:
                continue
            if s < lc - 1e-6 and interior_angle(-ux, -uy, cu[0], cu[1]) < limit:
                continue
            length = (la - d1) + t + (lc - s)
            cands.append((length, (round(px, 4), round(py, 4)), (round(qx, 4), round(qy, 4))))
    cands.sort(key=lambda c: c[0])
    seen = set()
    out = []
    for _l, P, Q in cands:
        if (P, Q) in seen:
            continue
        seen.add((P, Q))
        out.append((P, Q))
    return out


def tee_cut_candidates(V, A, C, min_angle: float, step: float, width: float = 0.0):
    """A branch V-C leaving a trunk V-A at an angle under ``min_angle``
    (a Y split, or a T whose end V lies on the trunk's body): the branch is
    moved to leave the trunk at P, on VA, along an octolinear cut P-Q to Q on
    VC, so it meets the trunk at ``min_angle`` or wider on both sides and
    bends at Q at ``min_angle`` or wider. P is at least ``width`` from V, so
    the new joint's copper is clear of the old one's. Yields (P, Q), the
    shortest branch first."""
    vx, vy = V
    ax, ay = A[0] - vx, A[1] - vy
    cx, cy = C[0] - vx, C[1] - vy
    la, lc = math.hypot(ax, ay), math.hypot(cx, cy)
    if la < 1e-9 or lc < 1e-9:
        return []
    au = (ax / la, ay / la)
    cu = (cx / lc, cy / lc)
    limit = float(min_angle) - ANGLE_TOL_DEG
    dirs = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]
    dirs += [(-au[1], au[0]), (au[1], -au[0])]     # square off a trunk off the 45 grid
    cands = []
    d1 = max(width, step)
    while d1 < la - 1e-6:
        px, py = vx + au[0] * d1, vy + au[1] * d1
        for ux, uy in dirs:
            if (interior_angle(au[0], au[1], ux, uy) < limit
                    or interior_angle(-au[0], -au[1], ux, uy) < limit):
                continue
            det = ux * (-cu[1]) - uy * (-cu[0])
            if abs(det) < 1e-12:
                continue
            rx, ry = vx - px, vy - py
            t = (rx * (-cu[1]) - ry * (-cu[0])) / det
            s = (ux * ry - uy * rx) / det
            if t <= 1e-6 or s <= 1e-6 or s > lc + 1e-6:
                continue
            if s < lc - 1e-6 and interior_angle(-ux, -uy, cu[0], cu[1]) < limit:
                continue
            qx, qy = vx + cu[0] * s, vy + cu[1] * s
            cands.append((t + (lc - s), (round(px, 4), round(py, 4)),
                          (round(qx, 4), round(qy, 4))))
        d1 += step
    cands.sort(key=lambda c: c[0])
    seen, out = set(), []
    for _l, P, Q in cands:
        if (P, Q) not in seen:
            seen.add((P, Q))
            out.append((P, Q))
    return out
