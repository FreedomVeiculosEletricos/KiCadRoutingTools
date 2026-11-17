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


# ---------------------------------------------------------------------------
# Route shape: zigzags, short legs and needless bends.
#
# A grid search lays its path on 45-degree moves, so what it leaves between
# two points is often not the path a designer draws: a run that keeps
# changing direction (a zigzag), a leg shorter than the track is wide
# between two bends, a bend the copper around it does not call for. A
# right-angle corner is allowed (only acute corners are a defect, graded by
# acute_corners); a 90-degree turn is never reported here. None of them is a DRC or connectivity
# fault, so nothing graded them; every one is a reviewer's finding:
#
#   * each bend is a discontinuity in the line's impedance and a place where
#     the etch rounds the inner corner and over-etches the outer one;
#   * a jog shorter than the track is wide is a lump of copper, not a line,
#     and narrows the gap to whatever it jogs toward;
#   * extra bends and detours add length (resistance, delay) and spend room
#     the board needs for other nets.
#
# Altium's and Allegro's glossing / "route smoothing" passes exist to remove
# exactly these: a straight or single-elbow octolinear line between two
# points wherever the copper around clears it. ``route_shape`` measures
# what is left. A run is followed along one net's tracks on one layer, from
# an anchor (a pad, a via, a branch or a free end) to the next, with
# collinear legs merged; its bends are the vertices in between.

#: A run that changes direction ZIGZAG_MIN_BENDS times in a row, each bend
#: closer than this to the one before along the run (mm), is a zigzag.
ZIGZAG_MAX_LEG_MM = 1.0
ZIGZAG_MIN_BENDS = 4
#: A leg between two bends shorter than this many track widths is a short leg.
SHORT_LEG_WIDTHS = 2.0
#: Bends turning this many degrees or more (right angles) are allowed and
#: never reported as needless.
RIGHT_ANGLE_DEG = 90.0
#: Windows of up to this many consecutive bends are tested for a shortcut.
NEEDLESS_MAX_WINDOW = 3


class ShapeFinding(NamedTuple):
    kind: str             # 'zigzag', 'short-leg', 'needless-bend'
    layer: str
    x: float
    y: float
    detail: str


class TrackRun(NamedTuple):
    layer: str
    pts: Tuple            # vertices, collinear ones merged: ((x, y), ...)
    widths: Tuple         # width of each leg (len(pts) - 1)
    ends: Tuple           # anchor kind at pts[0] and pts[-1]: 'pad', 'via',
                          # 'branch' or 'free'
    end_dirs: Tuple       # direction locked at each end (a pad's exit), or None


def _turn(ax, ay, bx, by, cx, cy) -> Tuple[float, float]:
    """(heading change in degrees, signed cross) of the path a -> b -> c."""
    ux, uy = bx - ax, by - ay
    vx, vy = cx - bx, cy - by
    lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
    if lu < 1e-12 or lv < 1e-12:
        return 0.0, 0.0
    c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
    return math.degrees(math.acos(c)), (ux * vy - uy * vx) / (lu * lv)


def track_runs(segs: Iterable, vias: Sequence = (), pads: Sequence = ()
               ) -> List[TrackRun]:
    """One net's tracks as runs between anchors, per layer, collinear legs
    merged. A vertex is interior when exactly two track ends meet there on
    that layer and neither the net's via nor its pad copper holds it."""
    by_layer = defaultdict(list)
    for s in segs:
        if getattr(s, 'graphic', False):
            continue
        if math.hypot(s.end_x - s.start_x, s.end_y - s.start_y) < 1e-6:
            continue
        by_layer[s.layer].append(s)
    runs: List[TrackRun] = []
    for layer, lsegs in by_layer.items():
        ends = defaultdict(list)
        for s in lsegs:
            ends[_key(s.start_x, s.start_y)].append(s)
            ends[_key(s.end_x, s.end_y)].append(s)
        covered = {}

        def kind(k):
            if k not in covered:
                x, y = k
                if any(math.hypot(x - v.x, y - v.y)
                       <= (getattr(v, 'size', 0.0) or 0.0) / 2.0 + 1e-3
                       for v in vias):
                    covered[k] = 'via'
                elif _covered(x, y, layer, (), pads):
                    covered[k] = 'pad'
                elif len(ends[k]) == 1:
                    covered[k] = 'free'
                elif len(ends[k]) == 2:
                    covered[k] = None
                else:
                    covered[k] = 'branch'
            return covered[k]

        used = set()
        starts = [k for k in ends if kind(k) is not None]
        for k0 in starts + [k for k in ends if kind(k) is None]:
            for s0 in ends[k0]:
                if id(s0) in used:
                    continue
                pts, widths, k, s = [k0], [], k0, s0
                while True:
                    used.add(id(s))
                    nk = (_key(s.end_x, s.end_y) if _key(s.start_x, s.start_y) == k
                          else _key(s.start_x, s.start_y))
                    pts.append(nk)
                    widths.append(s.width)
                    k = nk
                    if kind(k) is not None or k == k0:
                        break
                    nxt = [t for t in ends[k] if id(t) not in used]
                    if not nxt:
                        break
                    s = nxt[0]
                # merge collinear interior vertices
                mp, mw = [pts[0]], []
                for i in range(1, len(pts)):
                    if (i < len(pts) - 1 and len(mp) >= 1
                            and abs(widths[i - 1] - widths[i]) < 1e-6
                            and _turn(mp[-1][0], mp[-1][1], pts[i][0], pts[i][1],
                                      pts[i + 1][0], pts[i + 1][1])[0] < ANGLE_TOL_DEG):
                        continue
                    mp.append(pts[i])
                    mw.append(widths[i - 1])
                e0 = kind(mp[0]) or 'ring'
                e1 = kind(mp[-1]) or 'ring'
                runs.append(TrackRun(layer, tuple(mp), tuple(mw), (e0, e1),
                                     (None, None)))
    return runs


def route_shape(segs: Iterable, vias: Sequence = (), pads: Sequence = (),
                clears=None, zigzag_mm: float = ZIGZAG_MAX_LEG_MM,
                short_widths: float = SHORT_LEG_WIDTHS,
                zigzag_bends: int = ZIGZAG_MIN_BENDS) -> List[ShapeFinding]:
    """The shape findings of one net's tracks (see the block comment above).

    ``clears(x1, y1, x2, y2, layer, width)`` -> bool says whether a new
    track there keeps clear of foreign copper, the board edge and keep-outs;
    without it, V corners and needless bends are not graded (they need to
    know whether the copper allows the shortcut). A zigzag is reported once
    per chain of ``zigzag_bends`` or more consecutive bends each under
    ``zigzag_mm`` from the one before."""
    out: List[ShapeFinding] = []
    segs = [s for s in segs if not getattr(s, 'graphic', False)]
    for run in track_runs(segs, vias, pads):
        pts, ws, layer = run.pts, run.widths, run.layer
        n = len(pts)
        if n < 3:
            continue
        pinned = _pinned_legs(run, segs, vias, pads)
        turns = [None] * n
        for i in range(1, n - 1):
            turns[i] = _turn(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1],
                             pts[i + 1][0], pts[i + 1][1])
        bend = [t is not None and t[0] >= ANGLE_TOL_DEG for t in turns]
        # short legs: legs between two bends
        for i in range(1, n - 2):
            if not (bend[i] and bend[i + 1]):
                continue
            (ax, ay), (bx, by) = pts[i], pts[i + 1]
            leg = math.hypot(bx - ax, by - ay)
            w = ws[i]
            if leg < short_widths * w - 1e-6:
                out.append(ShapeFinding(
                    'short-leg', layer, (ax + bx) / 2.0, (ay + by) / 2.0,
                    f"{leg:.3f}mm leg between two bends, under "
                    f"{short_widths:g}x the {w:.3f}mm track width"))
        # zigzags: chains of consecutive bends, each under zigzag_mm from the
        # one before; one finding per chain of zigzag_bends or more
        i = 1
        while i < n - 1:
            if not bend[i]:
                i += 1
                continue
            j = i
            while (j + 1 < n - 1 and bend[j + 1]
                   and math.hypot(pts[j + 1][0] - pts[j][0],
                                  pts[j + 1][1] - pts[j][1]) < zigzag_mm - 1e-6):
                j += 1
            k = j - i + 1
            if k >= zigzag_bends:
                legs = [math.hypot(pts[q + 1][0] - pts[q][0], pts[q + 1][1] - pts[q][1])
                        for q in range(i, j)]
                m = (i + j) // 2
                out.append(ShapeFinding(
                    'zigzag', layer, pts[m][0], pts[m][1],
                    f"{k} bends in a row, each under {zigzag_mm:g}mm from the "
                    f"last (longest gap {max(legs):.3f}mm)"))
            i = j + 1
        if clears is None:
            continue
        flagged = {}
        for i in range(1, n - 1):
            if not bend[i]:
                continue
            for k in range(1, NEEDLESS_MAX_WINDOW + 1):
                j = i + k                      # C = pts[j], window i..j-1
                if j > n - 1 or not all(bend[q] for q in range(i, j)):
                    break
                if any(L in pinned for L in range(i - 1, j)):
                    continue
                alt = _shortcut(run, i - 1, j, k, clears)
                if alt is None:
                    continue
                for q in range(i, j):
                    if q not in flagged:
                        flagged[q] = (k, alt)
        for q, (k, alt) in sorted(flagged.items()):
            (x, y) = pts[q]
            deg = turns[q][0]
            if deg >= RIGHT_ANGLE_DEG - ANGLE_TOL_DEG:
                continue              # a right angle is allowed
            out.append(ShapeFinding(
                'needless-bend', layer, x, y,
                f"{deg:.0f}-degree bend: {k} bend(s) here can go, "
                f"{'a straight line' if not alt else 'one 45-degree elbow'} "
                f"clears the copper around"))
    return out


def _pinned_legs(run: TrackRun, segs, vias, pads) -> set:
    """Legs of ``run`` that same-net copper touches between their ends -- a
    track ending on the leg's body (a T), a via, a pad: moving the leg would
    strand it, so no shortcut may replace it."""
    pts, ws, layer = run.pts, run.widths, run.layer
    pinned = set()
    run_keys = set(pts)
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        r = ws[i] / 2.0
        lo_x, hi_x = min(ax, bx) - r - 1.0, max(ax, bx) + r + 1.0
        lo_y, hi_y = min(ay, by) - r - 1.0, max(ay, by) + r + 1.0
        for s in segs:
            if s.layer != layer:
                continue
            for ex, ey in ((s.start_x, s.start_y), (s.end_x, s.end_y)):
                if not (lo_x <= ex <= hi_x and lo_y <= ey <= hi_y):
                    continue
                if _key(ex, ey) in run_keys:
                    continue
                d, t = _pt_seg(ex, ey, ax, ay, bx, by)
                if d <= r + s.width / 2.0 + TEE_TOL:
                    pinned.add(i)
        for v in vias:
            if not (lo_x <= v.x <= hi_x and lo_y <= v.y <= hi_y):
                continue
            d, t = _pt_seg(v.x, v.y, ax, ay, bx, by)
            if d <= r + (getattr(v, 'size', 0.0) or 0.0) / 2.0 and 0.0 < t < 1.0:
                pinned.add(i)
        if pads:
            from pad_entry import pad_depth
            for p in pads:
                if not (layer in p.layers or any('*' in L for L in p.layers)):
                    continue
                reach = math.hypot(p.size_x, p.size_y) / 2.0 + r
                d, t = _pt_seg(p.global_x, p.global_y, ax, ay, bx, by)
                if d > reach or not (0.0 < t < 1.0):
                    continue
                mx, my = ax + (bx - ax) * t, ay + (by - ay) * t
                try:
                    if pad_depth(p, mx, my) > -r:
                        pinned.add(i)
                except Exception:
                    pinned.add(i)
    return pinned


def _shortcut(run: TrackRun, a: int, c: int, k: int, clears):
    """An octolinear replacement for the ``k`` bends between run vertices
    ``a`` and ``c`` with fewer bends, that clears and makes no new defect:
    no acute corner where it meets the run, the direction a pad end leaves
    in kept, and no leg shorter than the track width allows. Returns the
    intermediate points ([] = straight) or None."""
    from pcb_modification import _octolinear_bends
    pts, ws, layer = run.pts, run.widths, run.layer
    w = max(ws[a:c])
    if any(abs(x - w) > 1e-6 for x in ws[a:c]):
        return None                           # a width change sits inside
    A, C = pts[a], pts[c]
    n = len(pts)
    for inter in _octolinear_bends(A, C):
        if len(inter) >= k:
            continue
        poly = [A] + list(inter) + [C]
        legs = [math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(poly, poly[1:])]
        if min(legs) < 1e-6:
            continue
        if len(poly) > 2 and min(legs) < SHORT_LEG_WIDTHS * w:
            continue
        ok = True
        # where the new path meets the rest of the run
        for end, vi, prev, newp in ((0, a, a - 1, poly[1]), (1, c, c + 1, poly[-2])):
            if (end == 0 and vi == 0) or (end == 1 and vi == n - 1):
                kind = run.ends[end]
                if kind in ('pad', 'branch'):
                    # the leg leaving the pad (or the joint) keeps its line
                    old = pts[1] if end == 0 else pts[-2]
                    ox, oy = old[0] - pts[vi][0], old[1] - pts[vi][1]
                    nx, ny = newp[0] - pts[vi][0], newp[1] - pts[vi][1]
                    if interior_angle(ox, oy, nx, ny) > ANGLE_TOL_DEG:
                        ok = False
                continue
            px, py = pts[prev]
            vx, vy = pts[vi]
            if interior_angle(px - vx, py - vy, newp[0] - vx, newp[1] - vy) \
                    < 90.0 - ANGLE_TOL_DEG:
                ok = False
        if not ok:
            continue
        if all(clears(p[0], p[1], q[0], q[1], layer, w)
               for p, q in zip(poly, poly[1:])):
            return list(inter)
    return None
