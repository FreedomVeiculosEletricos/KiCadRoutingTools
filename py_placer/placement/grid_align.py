"""Grid snap and row/column alignment of placed parts.

The anneal moves parts by continuous random steps, so every part lands at an
arbitrary coordinate: its pads sit off the router's grid, and parts a
designer would line up -- a column of resistors, a row of caps -- stand a
fraction of a millimetre apart. The router then lays its vias on the grid
and joins them to the off-grid pads with jogs.

After the anneal (and the decap repair), place_simple:

1. **snaps** each movable part so that most of its pads land on the grid
   (`--grid`, default PLACE_GRID_MM, route.py's default --grid-step): the
   origin goes to the nearest legal grid point (within SNAP_RINGS steps)
   plus the part's own pad phase, the offset modulo the grid that puts the
   greatest number of its pads on a grid point;
2. **aligns** neighbours: movable parts on the same face, at the same
   rotation modulo 180 degrees, whose centres stand within `--align-mm`
   (default ALIGN_MM) across a column (or a row) and whose bodies are no
   more than ALIGN_REACH_MM apart along it, are moved to one shared
   coordinate -- a fixed member's if the group has one, else the one that
   moves the parts least.

Every move is kept only where the part stays legal (the same check the
anneal's result is graded by) and the board's decap and claim limits do not
get worse; otherwise the part stays where it was. Edge-seated parts and
parts left illegal are not touched.
"""
import math
from collections import Counter

#: the placement grid, mm: route.py's default --grid-step (routing_defaults.GRID_STEP)
PLACE_GRID_MM = 0.1
#: the largest offset across a row/column that alignment closes, mm
ALIGN_MM = 0.5
#: how far apart along a row/column two parts may be and still be aligned, mm
ALIGN_REACH_MM = 2.0
_EPS = 1e-6


def pad_phase(offsets, grid):
    """(px, py): the origin offset modulo ``grid`` that puts the most pads
    (``offsets``: [(dx, dy, ...)] from the origin) on grid points. A part
    with no pad has phase (0, 0): its origin goes on the grid."""
    if not offsets or grid <= 0:
        return 0.0, 0.0

    def best(vals):
        c = Counter(round((-v) % grid, 4) % round(grid, 4) for v in vals)
        top = max(c.values())
        return min(p for p, n in c.items() if n == top)
    return best([o[0] for o in offsets]), best([o[1] for o in offsets])


#: how many grid steps around a part its snap may look for a legal pose
SNAP_RINGS = 2
#: how many times the parts left off the grid are tried again
SNAP_PASSES = 3


def snap_candidates(x, y, phase, grid, rings=SNAP_RINGS):
    """The grid poses for a part of pad ``phase`` within ``rings`` grid
    steps of the cell around (x, y), nearest first, as (x, y) rounded to
    1e-6 mm: the four corners of its cell first, then the rings beyond."""
    px, py = phase
    fx = math.floor((x - px) / grid) * grid + px
    fy = math.floor((y - py) / grid) * grid + py
    out = []
    for i in range(1 - rings, rings + 1):
        for j in range(1 - rings, rings + 1):
            out.append((round(fx + i * grid, 6), round(fy + j * grid, 6)))
    out.sort(key=lambda p: (math.hypot(p[0] - x, p[1] - y), p))
    return out


def on_grid(x, y, phase, grid, tol=1e-4):
    px, py = phase
    return all(abs(((v - p) / grid) - round((v - p) / grid)) * grid <= tol
               for v, p in ((x, px), (y, py)))


def alignment_groups(items, tol=ALIGN_MM, reach=ALIGN_REACH_MM):
    """Groups of parts to line up. ``items``: {ref: (cx, cy, rect, side,
    rot)}, rect = (x0, y0, x1, y1). Returns [(axis, [refs])]: axis 'x' for a
    column (the refs share an x), 'y' for a row. Two parts pair up when they
    are on one face at one rotation modulo 180, their centres are within
    ``tol`` across the axis, and their rects are no more than ``reach``
    apart along it; pairs chain into groups."""
    refs = sorted(items)
    out = []
    for axis in ('x', 'y'):
        a = 0 if axis == 'x' else 1
        parent = {r: r for r in refs}

        def find(r):
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            return r
        for i, r in enumerate(refs):
            cr = items[r]
            for q in refs[i + 1:]:
                cq = items[q]
                if cr[3] != cq[3] or abs((cr[4] - cq[4]) % 180.0) > 1e-3 \
                        and abs((cr[4] - cq[4]) % 180.0 - 180.0) > 1e-3:
                    continue
                d = abs(cr[a] - cq[a])
                if d > tol + _EPS:
                    continue
                r0, q0 = cr[2], cq[2]
                if axis == 'x':     # a column: stacked in y
                    gap = max(r0[1], q0[1]) - min(r0[3], q0[3])
                else:
                    gap = max(r0[0], q0[0]) - min(r0[2], q0[2])
                if gap > reach:
                    continue
                parent[find(r)] = find(q)
        comp = {}
        for r in refs:
            comp.setdefault(find(r), []).append(r)
        for members in comp.values():
            if len(members) < 2:
                continue
            vals = {items[m][a] for m in members}
            if max(vals) - min(vals) > _EPS:
                out.append((axis, sorted(members)))
    return out


def shared_coordinate(values, fixed=()):
    """The coordinate a group lines up on: a fixed member's (the first, by
    value), else the median of ``values`` snapped to one of them -- the
    member value that moves the group least in total."""
    if fixed:
        return sorted(fixed)[0]
    return min(values, key=lambda c: (sum(abs(v - c) for v in values), c))


def snap_and_align(parts, movable, fixed_refs, *, grid, align_mm, try_pose,
                   limits=None, sensitive=None, refusals=None, log=print):
    """Snap ``movable`` parts to the grid, then line up neighbours.

    ``parts``: {ref: part} with .x .y .rot .side, ._pad_offsets(rot), .rect();
    ``try_pose(ref, x, y)``: move the part there and return True when it is
    legal, else put it back and return False; ``limits()``: a comparable
    count of broken limits (decap, claim), which no step may raise;
    ``refusals``: {ref: why} the first refused pose of each part was
    illegal, filled by ``try_pose``, for the report;
    ``sensitive``: the refs those limits bind (None: every part); a snap of
    one of them is kept only where the limits do not rise.
    Returns the report: {'grid', 'align_mm', 'snapped', 'off_grid',
    'off_grid_why' ({ref: 'limit' or why the pose was illegal}),
    'aligned': [{'axis', 'at', 'refs'}], 'not_aligned'}."""
    rep = {'grid': grid, 'align_mm': align_mm, 'snapped': 0, 'off_grid': [],
           'off_grid_why': {},
           'aligned': [], 'not_aligned': []}
    base = limits() if limits else None
    if grid and grid > 0:
        snapped = 0
        todo = sorted(movable)
        # a part pressed against a neighbour may find room once the
        # neighbour has snapped: the parts left are tried again while any moves
        for _pass in range(SNAP_PASSES):
            left, why_of = [], {}
            for r in todo:
                P = parts[r]
                ph = pad_phase(P._pad_offsets(P.rot), grid)
                if on_grid(P.x, P.y, ph, grid):
                    continue
                x0, y0 = P.x, P.y
                why = 'illegal'
                if refusals is not None:
                    refusals.pop(r, None)
                for x, y in snap_candidates(x0, y0, ph, grid):
                    if not try_pose(r, x, y):
                        continue
                    # a part a decap or claim limit binds keeps the limits it had
                    if limits and (sensitive is None or r in sensitive) and limits() > base:
                        try_pose(r, x0, y0)
                        why = 'limit'
                        continue
                    snapped += 1
                    break
                else:
                    left.append(r)
                    why_of[r] = why
            done_now = len(todo) - len(left)
            todo = left
            if not left or not done_now:
                break
        for r in todo:
            rep['off_grid'].append(r)
            why = why_of[r]
            rep['off_grid_why'][r] = (why if why == 'limit' or not refusals
                                      else refusals.get(r) or why)
        rep['snapped'] = snapped
    if align_mm and align_mm > 0:
        allr = sorted(set(movable) | set(fixed_refs))
        items = {}
        for r in allr:
            P = parts[r]
            items[r] = (P.x, P.y, P.rect(), getattr(P, 'side', None), P.rot % 360.0)
        for axis, members in alignment_groups(items, tol=align_mm):
            a = 0 if axis == 'x' else 1
            fixed = [items[m][a] for m in members if m not in movable]
            if len(set(round(v, 6) for v in fixed)) > 1:
                continue            # two fixed parts disagree: nothing to line up on
            at = shared_coordinate([items[m][a] for m in members], fixed)
            moved, left = [], []
            undo = []
            for m in members:
                if m not in movable or abs(items[m][a] - at) < _EPS:
                    continue
                P = parts[m]
                x0, y0 = P.x, P.y
                x, y = (at, y0) if axis == 'x' else (x0, at)
                if try_pose(m, x, y):
                    moved.append(m)
                    undo.append((m, x0, y0))
                else:
                    left.append(m)
            if moved and limits and limits() > base:
                for m, x0, y0 in reversed(undo):
                    try_pose(m, x0, y0)
                left += moved
                moved = []
            if moved:
                rep['aligned'].append({'axis': axis, 'at': round(at, 4),
                                       'refs': sorted(moved)})
                for m in moved:
                    P = parts[m]
                    items[m] = (P.x, P.y, P.rect(), items[m][3], items[m][4])
            if left:
                rep['not_aligned'].extend(sorted(left))
    log(f"  grid: {rep['snapped']} part(s) snapped to the {grid:g} mm grid"
        + ''.join(f", {len(v)} left off it ({k}): {', '.join(v)}" for k, v in (
            ('no legal grid pose', [f"{r} ({rep['off_grid_why'][r]})" for r in rep['off_grid']
                                    if rep['off_grid_why'][r] != 'limit']),
            ('a decap or claim limit would break', [r for r in rep['off_grid']
                                                    if rep['off_grid_why'][r] == 'limit'])) if v)
        + (f"; {sum(len(g['refs']) for g in rep['aligned'])} part(s) lined up "
           f"in {len(rep['aligned'])} row(s)/column(s)" if rep['aligned'] else '')
        + (f"; not lined up (no legal pose): {', '.join(sorted(set(rep['not_aligned'])))}"
           if rep['not_aligned'] else ''))
    return rep
