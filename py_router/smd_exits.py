"""SMD pad exits laid by the router itself (Altium's SMD Entry / SMD To Corner).

A grid route that ends on an SMD pad's cells leaves the pad wherever the
search found it cheapest: through a corner, at 45 degrees across an edge, or
bending on the outline (pad_entry.grade_pad_exit). Altium's router instead
starts every pad exit at the pad and runs it straight out through an allowed
edge before the first bend. This module does the same, as the TERMINAL of the
search rather than a repair after it:

  * ``exit_mouths`` builds, for one SMD pad, every compliant exit: a straight
    run from an ANCHOR inside the pad -- the pad centre projected onto the
    grid line nearest the pad's axis, deep enough for the pad entry rule --
    perpendicular out through an allowed edge (the ends always, the sides
    with SMD_ENTRY_SIDE), to a MOUTH grid cell at least the run's SMD To
    Corner distance past the outline. An exit is offered only when the run
    grades clean (grade_pad_exit), its anchor is a full-depth entry, its
    mouth cell is free and the run clears every foreign object exactly
    (pcb_modification._connector_clear) and every keep-out;
  * the route searches from / to the MOUTHS instead of the pad's cells, each
    carrying its heading (grid_router 0.23.8 source_headings /
    target_headings): the first move off a mouth, and the move into one,
    turns at most 90 degrees, and each mouth costs its exit's length -- so
    the search picks the exit, end or side, by the cost of the WHOLE route,
    not by where the target lies;
  * the pad itself is fenced for those searches (pad_fence_cells): the
    route reaches it only through a mouth, never across its copper;
  * the route welds to the mouth's anchor, so the straight exit is the
    route's own terminal leg; on a narrow off-grid pad, whose nearest grid
    line is too shallow, the exit runs on the pad's axis instead and the
    route's end on the mouth cell is drawn at the mouth point there
    (mouth_point), so the leg is exactly perpendicular too.

A pad with no compliant exit keeps its plain cells (and check_weird reports
what the route then lays); a search the mouths leave without a path is
retried on the plain cells, so the exits never cost a net.
"""
from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Tuple

#: Rust ORTHO_COST: the g of one orthogonal grid step.
_ORTHO_COST = 1000

#: The mouths of the searches in flight: (gx, gy, layer_idx) -> (hx, hy,
#: extra_g), the heading pointing OUT of the pad. Installed by
#: ``active_mouths`` for one net's search and read by ``heading_rows``.
_ACTIVE: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}


#: What the run's searches did with the exits (summary_line):
#: pads given mouths, pads with no compliant exit, searches retried plain.
STATS = {'pads': 0, 'no_exit': 0, 'fell_back': 0}


def note_fallback():
    """A search the mouths left without a path was retried on the pad cells."""
    STATS['fell_back'] += 1


def summary_line() -> str:
    """One log line for the end of the run ('' when no SMD pad was met)."""
    if not (STATS['pads'] or STATS['no_exit']):
        return ''
    return (f"SMD exits (smd_exits): {STATS['pads']} pad terminal(s) searched through "
            f"their straight exits, {STATS['no_exit']} with no exit that fits "
            f"(searched on the pad), {STATS['fell_back']} search(es) the exits left "
            f"without a path retried on the pad")


#: Whether a swap_terminals inside the current active_mouths block swapped
#: any pad terminal for mouths (the caller retries without them on failure).
_SWAPPED = [False]


#: The pad fences of the active_mouths blocks in flight, innermost last:
#: (obstacles, cells) as handed to add_blocked_cells_batch.
_FENCES: List[tuple] = []

#: The mouths of the last active_mouths block that ended (diagnostics).
LAST_BLOCK: List[dict] = [{}]


def pad_fence_cells(pad, layer_idx, width, step) -> List[Tuple[int, int, int]]:
    """The grid cells of ``layer_idx`` a ``width`` track centred on would
    touch ``pad``'s copper: the pad outline grown by half the width --
    the obstacle map's BASE width (config.track_width): the router sweeps a
    wider net's extra half-width around every blocked cell itself
    (track_margin), so a fence at the net's own width would count that extra
    twice and wall the mouths of every wide net in. A pad
    searched through its mouths is fenced with them, so the route reaches
    the pad only along an exit -- not across its copper to a mouth from the
    far side, nor grazing its outline on the way."""
    from pad_entry import pad_depth
    reach = width / 2.0
    r = math.hypot(pad.size_x, pad.size_y) / 2.0 + reach
    cx, cy = pad.global_x, pad.global_y
    out = []
    for gx in range(int(math.floor((cx - r) / step)), int(math.ceil((cx + r) / step)) + 1):
        for gy in range(int(math.floor((cy - r) / step)), int(math.ceil((cy + r) / step)) + 1):
            if pad_depth(pad, gx * step, gy * step) > -reach + 1e-6:
                out.append((gx, gy, layer_idx))
    return out


def fence_pad(obstacles, pad, layer_idx, width, step) -> None:
    """Block ``pad``'s fence (pad_fence_cells) on ``obstacles`` until the
    enclosing active_mouths / fenced block ends. Mouth cells are source /
    target cells, which a block never closes."""
    if obstacles is None or not hasattr(obstacles, 'add_blocked_cells_batch'):
        return
    cells = pad_fence_cells(pad, layer_idx, width, step)
    if not cells:
        return
    import numpy as np
    arr = np.asarray(cells, dtype=np.int32)
    obstacles.add_blocked_cells_batch(arr)
    _FENCES.append((obstacles, arr))


def _drop_fences(n0: int) -> None:
    while len(_FENCES) > n0:
        obs, arr = _FENCES.pop()
        obs.remove_blocked_cells_batch(arr)


@contextlib.contextmanager
def fenced(fences=()):
    """Fence each ``(obstacles, pad, layer_idx, width, step)`` for the block."""
    n0 = len(_FENCES)
    try:
        for f in fences or ():
            fence_pad(*f)
        yield
    finally:
        _drop_fences(n0)


@contextlib.contextmanager
def active_mouths(mouths, fences=()):
    """Install ``mouths`` ({cell: (hx, hy, extra_g)}) and the pad ``fences``
    (fence_pad arguments) for the searches inside the block, restoring what
    was there before. swap_terminals inside the block adds its own."""
    saved = dict(_ACTIVE)
    saved_flag = _SWAPPED[0]
    n0 = len(_FENCES)
    _ACTIVE.update(mouths or {})
    _SWAPPED[0] = False
    try:
        for f in fences or ():
            fence_pad(*f)
        yield
    finally:
        _drop_fences(n0)
        LAST_BLOCK[:] = [dict(_ACTIVE)]
        _ACTIVE.clear()
        _ACTIVE.update(saved)
        _SWAPPED[0] = saved_flag


#: May a search the exits leave without a path be retried on the plain pad
#: cells? Not while the run's main stages route (no_plain_fallback): there a
#: failure reaches the rip-up ladder, which frees the exit another net took;
#: the later stages (rescue, reconciliation) may still land on the pad.
_FALLBACK = [True]


def fallback_allowed() -> bool:
    return _FALLBACK[0] and not _STRICT[0]


#: Strict pad exits (RoutingConfig.pad_exit_strict, the default): every pad
#: the exits can grade -- SMD and through-hole, every net -- is reached ONLY
#: through a straight exit: perpendicular out of a flat side, or at 45
#: degrees through a corner on its bisector (pad_entry.corner_exits, the
#: default), never at any other angle or place, never bending before SMD To
#: Corner. A
#: pad with no such exit is no terminal and a search its exits leave without
#: a path is not retried on the pad: the connection stays open instead.
#: Off: the historical behaviour (SMD pads only, plain-pad retry, skip list).
def _strict_env_allows() -> bool:
    """KICAD_STRICT_PAD_EXITS=0 turns strict exits off for the process (the
    tests that pin the non-strict mechanisms -- a via in a pad, the plain-pad
    retry, the own-pad lift -- set it; a run passes --no-strict-pad-exits)."""
    import os
    return os.environ.get('KICAD_STRICT_PAD_EXITS', '1').strip().lower() \
        not in ('0', 'false', 'off', 'no')


_STRICT = [_strict_env_allows()]


def set_strict(on) -> bool:
    """Set the run's strict-exit mode; returns the previous one."""
    prev = _STRICT[0]
    _STRICT[0] = bool(on) and _strict_env_allows()
    return prev


def strict() -> bool:
    return _STRICT[0]


def exit_gradable(pad, layer: str) -> bool:
    """Does the exit rule govern ``pad`` on ``layer``? SMD pads always;
    with strict exits every pad pad_entry grades (through-hole too)."""
    from pad_entry import smd_gradable, gradable_pad
    if _STRICT[0]:
        return gradable_pad(pad, layer)
    return smd_gradable(pad, layer)


def no_plain_fallback(fn):
    """``fn`` run with the plain-pad retry off (see _FALLBACK)."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        prev = _FALLBACK[0]
        _FALLBACK[0] = False
        try:
            return fn(*a, **k)
        finally:
            _FALLBACK[0] = prev
    return run


def swapped() -> bool:
    """Did a swap_terminals in the current active_mouths block lay mouths?"""
    return _SWAPPED[0]


#: pad_exit_skip_net_ids per board (it reads the board's rules and records).
_SKIP_CACHE: Dict[int, frozenset] = {}


def skip_net_ids(pcb_data) -> frozenset:
    """Nets whose pad exits are the spec (cleanup_pipeline.pad_exit_skip_net_ids):
    no mouths are laid for them."""
    if _STRICT[0]:
        return frozenset()      # strict exits govern every net
    key = id(pcb_data)
    if key not in _SKIP_CACHE:
        try:
            from cleanup_pipeline import pad_exit_skip_net_ids
            _SKIP_CACHE[key] = frozenset(pad_exit_skip_net_ids(pcb_data))
        except Exception:                                    # noqa: BLE001
            _SKIP_CACHE[key] = frozenset()
    return _SKIP_CACHE[key]


def swap_terminals(sources, targets, pcb_data, net_id, config, obstacles,
                   layer_names):
    """smd_exit_rows on both ends of one search, the mouths installed for the
    searches of the enclosing active_mouths block. Returns (sources,
    targets)."""
    skip = skip_net_ids(pcb_data)
    s2, ms = smd_exit_rows(sources, pcb_data, net_id, config, obstacles,
                           layer_names, skip)
    t2, mt = smd_exit_rows(targets, pcb_data, net_id, config, obstacles,
                           layer_names, skip)
    if ms or mt:
        _ACTIVE.update(ms)
        _ACTIVE.update(mt)
        _SWAPPED[0] = True
    return s2, t2


def heading_rows(sources, targets) -> dict:
    """route_multi / route_with_frontier keyword arguments for one search:
    the active mouths among ``sources`` (heading out of the pad, the way the
    track runs on from it) and among ``targets`` (heading into the pad, the
    way the track continues into it). {} when no mouth is in play."""
    if not _ACTIVE:
        return {}
    sh = [(c[0], c[1], c[2], h[0], h[1], h[2])
          for c in sources for h in (_ACTIVE.get((c[0], c[1], c[2])),) if h]
    th = [(c[0], c[1], c[2], -h[0], -h[1], h[2])
          for c in targets for h in (_ACTIVE.get((c[0], c[1], c[2])),) if h]
    out = {}
    if sh:
        out['source_headings'] = sh
    if th:
        out['target_headings'] = th
    return out


def _octant(dx, dy):
    """(sx, sy) of a unit direction that is octolinear to 0.5 degree, else
    None (a pad at an odd rotation has no exit the grid can carry)."""
    a = math.degrees(math.atan2(dy, dx))
    k = round(a / 45.0)
    if abs(a - 45.0 * k) > 0.5:
        return None
    return {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0),
            -1: (1, -1), -2: (0, -1), -3: (-1, -1), -4: (-1, 0)}[k]


def _exit_dirs(pad):
    """Board-frame ((ux, uy) unit, (sx, sy) octant, extent) of each allowed
    exit of ``pad``: extent is the distance from the centre to the outline
    along it. With corner exits (pad_entry.corner_exits) the four 45-degree
    exits through the corners follow the edge exits, each with its pad-frame
    corner signs in place of the extent: ``(..., ('corner', ex, ey))``."""
    from pad_entry import (allowed_exits, to_board_dir, _round_pad,
                           corner_exits)
    out = []
    corners = corner_exits()
    if _round_pad(pad):
        r = pad.size_x / 2.0
        dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
        if not _STRICT[0] and not corners:
            dirs += ((1, 1), (-1, 1), (-1, -1), (1, -1))
        for sx, sy in dirs:
            n = math.hypot(sx, sy)
            out.append(((sx / n, sy / n), (sx, sy), r))
    else:
        for nx, ny in allowed_exits(pad):
            ux, uy = to_board_dir(pad, nx, ny)
            oc = _octant(ux, uy)
            if oc is None:
                continue
            extent = pad.size_x / 2.0 if abs(nx) > 0.5 else pad.size_y / 2.0
            out.append(((ux, uy), oc, extent))
    if corners:
        r2 = math.sqrt(0.5)
        for ex, ey in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            ux, uy = to_board_dir(pad, ex * r2, ey * r2)
            oc = _octant(ux, uy)
            if oc is None or not (oc[0] and oc[1]):
                continue        # a pad rotated off 45 degrees: no grid diagonal
            out.append(((ux, uy), oc, ('corner', ex, ey)))
    return out


#: Extra g per grid step a lane sits off the pad's axis (exit_mouths): a
#: twentieth of a step, a tie-break toward the centred exit, never a reason
#: to jog.
_LANE_COST = 50


def _nearest_lane(pad, sx, step):
    """The grid line nearest the pad's axis an exit along x (sx != 0) or y
    runs on."""
    return (round(pad.global_y / step) * step if sx != 0
            else round(pad.global_x / step) * step)


def _exit_lanes(pad, step, need):
    """_exit_dirs, each orthogonal exit repeated per grid LANE it may run
    on: ``((ux, uy), (sx, sy), extent, lane)``, lane being the exit line's
    board y (an exit along x) or x (along y). The nearest lane comes first;
    with strict exits the others the pad's flat side spans -- a lane whose
    full-depth anchor the pad holds -- follow, nearest first. A diagonal
    exit has the one lane None."""
    out = []
    for (ux, uy), (sx, sy), extent in _exit_dirs(pad):
        if sx != 0 and sy != 0:
            if isinstance(extent, tuple):
                # a corner exit: on the grid diagonal nearest the corner's
                # bisector (its deepest point p0 projected onto that line)
                from pad_entry import corner_line, to_board_dir
                (px, py), _d = corner_line(pad, extent[1], extent[2])
                bx, by = to_board_dir(pad, px, py)
                p0x, p0y = pad.global_x + bx, pad.global_y + by
                # the 45-degree grid lines of heading (sx, sy): x - sx*sy*y = c
                c = round((p0x - sx * sy * p0y) / step) * step
                tt = (p0x - sx * sy * p0y - c) / 2.0
                ax_, ay_ = p0x - tt, p0y + sx * sy * tt
                out.append(((ux, uy), (sx, sy), extent, ('corner', ax_, ay_)))
                continue
            out.append(((ux, uy), (sx, sy), extent, None))
            continue
        c = pad.global_y if sx != 0 else pad.global_x
        from pad_entry import _round_pad
        if _STRICT[0] and _round_pad(pad):
            # a round pad is left radially: on the grid line through its
            # centre when one passes within SMD_ENTRY_OVERHANG_MM of it (an
            # on-grid exit, no tilted leg after it), else on the centre line
            # exactly (the route's end drawn at the mouth point)
            from routing_defaults import SMD_ENTRY_OVERHANG_MM
            g = _nearest_lane(pad, sx, step)
            if abs(g - c) <= SMD_ENTRY_OVERHANG_MM + 1e-9:
                out.append(((ux, uy), (sx, sy), extent, round(g, 6)))
            else:
                out.append(((ux, uy), (sx, sy), extent, round(c, 6)))
            continue
        base = _nearest_lane(pad, sx, step)
        lanes = [base]
        if _STRICT[0] and abs(pad.rotation if hasattr(pad, 'rotation') else 0) % 90 < 1e-6:
            from pad_entry import pad_depth
            for k in range(1, 12):
                for v in (base + k * step, base - k * step):
                    px, py = (pad.global_x, v) if sx != 0 else (v, pad.global_y)
                    if pad_depth(pad, px, py) >= need:
                        lanes.append(v)
        lanes.sort(key=lambda v: abs(v - c))
        for v in lanes:
            out.append(((ux, uy), (sx, sy), extent, round(v, 6)))
    return out


def exit_mouths(pad, layer_name, layer_idx, width, step, pcb_data, net_id,
                config, obstacles=None) -> List[tuple]:
    """Every compliant straight exit of SMD ``pad`` on ``layer_name`` for a
    ``width`` track on a ``step`` grid, as rows ``(gx, gy, layer_idx,
    anchor_x, anchor_y, sx, sy, extra_g)``: the mouth cell, the anchor the
    route welds to, the outward octant heading and the exit's cost. [] when
    the pad is not graded or no exit fits."""
    from pad_entry import (smd_gradable, pad_depth, required_depth,
                           grade_pad_exit, exit_distance, DEPTH_TOL)
    if not exit_gradable(pad, layer_name):
        return []
    host = _host_pad(pad, layer_name, pcb_data)
    if host is not None:
        # a pad inside a larger same-net pad (an exposed pad's paste or via
        # sub-pads) is that pad's copper: it is left through the host's exits
        pad = host
    # a pad holding same-net sub-pads has no corner exit: its 45-degree leg
    # would cross the sub-pads at an angle, each of them graded as a pad
    hosts = _hosts_sub_pads(pad, layer_name, pcb_data)
    from pcb_modification import _connector_clear
    from single_ended_routing import _keepout_gate, _track_refused
    cx, cy = pad.global_x, pad.global_y
    need = required_depth(pad, width)
    run = exit_distance(width)
    clr = config.layer_clearance(layer_name, config.obstacle_clearance(net_id),
                                 net_id)
    ncl = getattr(config, 'net_clearances', None) or None
    gate = _keepout_gate(pcb_data)
    rows = []
    seen = set()
    for (ux, uy), (sx, sy), extent, lane in _exit_lanes(pad, step, need):
        diag = sx != 0 and sy != 0
        cstep = step * (math.sqrt(2.0) if diag else 1.0)   # one grid move along d
        # An orthogonal exit runs along a grid LANE across the pad's flat
        # side (_exit_lanes: the line nearest the axis first, then every
        # other grid line the side's flat part carries a full-depth exit
        # on), so the route leaves on the lane it continues along instead
        # of stepping across right past the mouth; a lane that is not deep
        # enough falls back to the pad's axis from the centre, the route's
        # end on the mouth cell drawn at the mouth POINT -- the cell moved
        # across onto the axis (mouth_point; at most half a grid step) -- so
        # the exit is still exactly perpendicular. A corner exit runs at 45
        # degrees on the grid diagonal nearest the corner's bisector
        # (_exit_lanes), from that bisector's deepest point; the legacy
        # diagonal of a round pad (corner exits off, non-strict) keeps the
        # grid diagonal nearest the centre.
        corner = isinstance(lane, tuple)
        if corner and hosts:
            continue
        if corner:
            ax, ay = lane[1], lane[2]
        elif not diag:
            ax, ay = (cx, lane) if sx != 0 else (lane, cy)
            if pad_depth(pad, ax, ay) < need - DEPTH_TOL:
                if lane != _nearest_lane(pad, sx, step):
                    continue
                ax, ay = cx, cy
        else:
            gx0, gy0 = round(cx / step) * step, round(cy / step) * step
            # project the centre onto the diagonal through (gx0, gy0)
            t = ((cx - gx0) * ux + (cy - gy0) * uy)
            ax, ay = gx0 + t * ux, gy0 + t * uy
        if pad_depth(pad, ax, ay) < need - DEPTH_TOL:
            continue            # the grid line is too far off the axis
        # the first grid point along d at least `run` past the outline
        if corner:
            lo, hi = 0.0, 2.0 * math.hypot(pad.size_x, pad.size_y)
            for _ in range(40):         # the outline crossing along d
                mid = (lo + hi) / 2.0
                if pad_depth(pad, ax + ux * mid, ay + uy * mid) > 0:
                    lo = mid
                else:
                    hi = mid
            along = hi + run
        else:
            along = extent + run - ((ax - cx) * ux + (ay - cy) * uy)
        if corner:
            # grid points on the anchor's 45-degree grid line, from the one
            # nearest the anchor outward
            mx = round(ax / step) * step
            my = ay + (mx - ax) * sx * sy
            while ((mx - ax) * ux + (my - ay) * uy) < along - 1e-6:
                mx, my = mx + sx * step, my + sy * step
        elif not diag:
            if sx != 0:
                mx = (math.ceil((ax + sx * along) / step - 1e-6) * step if sx > 0
                      else math.floor((ax + sx * along) / step + 1e-6) * step)
                my = ay
            else:
                my = (math.ceil((ay + sy * along) / step - 1e-6) * step if sy > 0
                      else math.floor((ay + sy * along) / step + 1e-6) * step)
                mx = ax
        else:
            k = math.ceil(along / cstep - 1e-6)
            base_x, base_y = round(cx / step) * step, round(cy / step) * step
            # the grid point on the diagonal nearest the anchor, then k moves
            mx, my = base_x + sx * step * k, base_y + sy * step * k
            while ((mx - ax) * ux + (my - ay) * uy) < along - 1e-6:
                mx, my = mx + sx * step, my + sy * step
        gx, gy = int(round(mx / step)), int(round(my / step))
        # the mouth point: the grid cell, moved across onto the exit's line
        # (the cell itself when the exit runs on a grid line)
        if not diag:
            mx, my = (gx * step, ay) if sx != 0 else (ax, gy * step)
        else:
            mx, my = gx * step, gy * step
        if (gx, gy) in seen:
            continue
        if obstacles is not None and obstacles.is_blocked(gx, gy, layer_idx):
            continue
        if grade_pad_exit(pad, [(ax, ay), (mx, my),
                                (mx + sx * step, my + sy * step)], width):
            continue
        if not _connector_clear(ax, ay, mx, my, width, layer_name, net_id,
                                pcb_data, clr, net_clearances=ncl):
            continue
        # the net's own protected sense copper (#1279) is foreign to the
        # leg too -- the measure the terminal neck applies to the route's
        # terminal (single_ended_routing._terminal_room)
        from sense_branch import terminal_sense_dist
        if (terminal_sense_dist(pcb_data, net_id, ax, ay, mx, my, layer_name,
                                width / 2.0) - width / 2.0 < clr - 1e-9):
            continue
        if _track_refused(gate, net_id, config, layer_name, (ax, ay), (mx, my)):
            continue
        if _leg_meets_other_pad(pad, layer_name, ax, ay, mx, my, width, pcb_data):
            continue
        seen.add((gx, gy))
        g = int(round(math.hypot(mx - ax, my - ay) / step * _ORTHO_COST))
        # off-axis lanes cost a little more, so a tie keeps the centred exit
        if not diag:
            g += int(round(abs((ay - cy) if sx != 0 else (ax - cx)) / step
                           * _LANE_COST))
        ax, ay = round(ax, 6), round(ay, 6)
        _MOUTH_PT[_mouth_key(gx, gy, layer_idx, ax, ay)] = (round(mx, 6), round(my, 6))
        rows.append((gx, gy, layer_idx, ax, ay, sx, sy, g))
    return rows


def _host_pad(pad, layer_name, pcb_data):
    """The larger same-net pad on ``layer_name`` whose copper holds ``pad``'s
    centre and outline box (an exposed pad around its sub-pads), or None."""
    if pcb_data is None or not getattr(pad, 'net_id', 0):
        return None
    from pad_entry import pad_depth
    best = None
    for q in pcb_data.pads_by_net.get(pad.net_id, []) or ():
        if q is pad or not (layer_name in q.layers
                            or any('*' in L for L in q.layers)):
            continue
        if q.size_x * q.size_y <= pad.size_x * pad.size_y + 1e-9:
            continue
        hx, hy = pad.size_x / 2.0, pad.size_y / 2.0
        if all(pad_depth(q, pad.global_x + ex * hx, pad.global_y + ey * hy) >= -1e-6
               for ex in (-1, 1) for ey in (-1, 1)) or (
                pad_depth(q, pad.global_x, pad.global_y) > min(hx, hy)):
            if best is None or q.size_x * q.size_y > best.size_x * best.size_y:
                best = q
    return best


def _hosts_sub_pads(pad, layer_name, pcb_data) -> bool:
    """Does ``pad`` hold another same-net pad of ``layer_name`` (its host,
    _host_pad, is ``pad``)?"""
    if pcb_data is None or not getattr(pad, 'net_id', 0):
        return False
    return any(q is not pad and _host_pad(q, layer_name, pcb_data) is pad
               for q in pcb_data.pads_by_net.get(pad.net_id, []) or ())


def _leg_meets_other_pad(pad, layer_name, ax, ay, mx, my, width, pcb_data):
    """Does the exit leg anchor -> mouth touch any OTHER pad's copper on its
    layer -- the net's own included? A leg into a neighbouring same-net pad
    (a side exit across a 0.5 mm pitch gap) is no exit: it joins the two pads
    through the gap between them instead of leaving either straight out.
    Foreign copper is _connector_clear's business."""
    from pad_entry import segment_pad_depth, pad_depth
    reach = width / 2.0 + math.hypot(mx - ax, my - ay)
    for p in _all_pads(pcb_data):
        if p is pad or not (layer_name in p.layers
                            or any('*' in L for L in p.layers)):
            continue
        if (p.net_id == pad.net_id and p.net_id
                and pad_depth(p, pad.global_x, pad.global_y) > 0.0):
            continue    # the same pad stacked (a USB-C A4/B9): one copper
        r = math.hypot(p.size_x, p.size_y) / 2.0 + reach
        if abs(p.global_x - ax) > r or abs(p.global_y - ay) > r:
            continue
        if segment_pad_depth(p, ax, ay, mx, my) > -width / 2.0:
            return True
    return False


_PADS_CACHE: Dict[int, tuple] = {}


def _all_pads(pcb_data):
    """Every pad of the board (pads_by_net flattened; cached per board)."""
    pbn = getattr(pcb_data, 'pads_by_net', None) or {}
    key = id(pcb_data)
    hit = _PADS_CACHE.get(key)
    if hit is None or hit[0] is not pbn:
        out = []
        for ps in pbn.values():
            out.extend(ps)
        hit = (pbn, out)
        _PADS_CACHE[key] = hit
    return hit[1]


#: (gx, gy, layer_idx, anchor) -> the exit's mouth point (exit_mouths): where
#: a route ending on that mouth cell and welded to that anchor is drawn.
_MOUTH_PT: Dict[tuple, Tuple[float, float]] = {}


def _mouth_key(gx, gy, li, ax, ay):
    return (gx, gy, li, round(ax, 4), round(ay, 4))


def mouth_point(cell, anchor):
    """The mouth point of the exit ``cell`` -> ``anchor`` (x, y), or None when
    the pair is not an exit's mouth and anchor."""
    if anchor is None:
        return None
    return _MOUTH_PT.get(_mouth_key(cell[0], cell[1], cell[2], anchor[0], anchor[1]))


def smd_exit_rows(rows, pcb_data, net_id, config, obstacles, layer_names,
                  skip_net_ids=frozenset()):
    """Swap each SMD-pad terminal row ``(gx, gy, layer_idx, x, y)`` of
    ``rows`` -- one whose (x, y) is a gradable SMD pad's centre -- for that
    pad's exit mouths (exit_mouths). Returns ``(rows, mouths)``: the new rows
    (non-pad rows, and pads with no compliant exit, unchanged) and the
    {cell: (sx, sy, extra_g)} the search installs (active_mouths). ``mouths``
    is empty when nothing was swapped."""
    if not rows or net_id in skip_net_ids or not getattr(config, 'smd_exit_terminals', True):
        return list(rows or ()), {}
    pads = pcb_data.pads_by_net.get(net_id, []) or []
    by_pt = {}
    for p in pads:
        by_pt.setdefault((round(p.global_x, 4), round(p.global_y, 4)), []).append(p)
    step = config.grid_step
    out, mouths, done = [], {}, {}
    dropped = []
    for r in rows:
        cand = by_pt.get((round(r[3], 4), round(r[4], 4)))
        layer = layer_names[r[2]] if r[2] < len(layer_names) else None
        pad = next((p for p in (cand or ()) if layer and
                    (layer in p.layers or any('*' in L for L in p.layers))), None)
        if pad is None or layer is None:
            out.append(r)
            continue
        key = (id(pad), r[2])
        if key not in done:
            w = config.get_net_track_width(net_id, layer)
            done[key] = exit_mouths(pad, layer, r[2], w, step, pcb_data,
                                    net_id, config, obstacles)
        ms = done[key]
        if not ms:
            gradable = exit_gradable(pad, layer)
            if not done.get(('counted',) + key):
                done[('counted',) + key] = True
                if gradable:
                    STATS['no_exit'] += 1
            if gradable and _STRICT[0]:
                dropped.append((pad, r[2]))
                continue        # strict: no compliant exit, no terminal here
            out.append(r)
            continue
        if done.get(('emitted',) + key):
            continue
        done[('emitted',) + key] = True
        STATS['pads'] += 1
        fence_pad(obstacles, pad, r[2], config.track_width, step)
        for gx, gy, li, ax, ay, sx, sy, g in ms:
            out.append((gx, gy, li, ax, ay))
            mouths[(gx, gy, li)] = (sx, sy, g)
    # A swapped pad keeps no other launch cell on its copper (a #479 seeded
    # on-pad cell would let the route end in the pad and skip its exits).
    swapped_pads = [(pad_, li_) for (k0, k1), ms_ in
                    ((k, v) for k, v in done.items()
                     if k[0] not in ('emitted', 'counted'))
                    for pad_ in (next((p for p in pads if id(p) == k0), None),)
                    for li_ in (k1,) if ms_ and pad_ is not None]
    if _STRICT[0] and obstacles is not None:
        for f in net_pad_fences(pcb_data, net_id, obstacles, config, layer_names):
            fence_pad(*f)
    swapped_pads = swapped_pads + dropped
    if swapped_pads:
        from pad_entry import pad_depth
        out = [r for r in out
               if (r[0], r[1], r[2]) in mouths
               or not any(r[2] == li_ and pad_depth(pad_, r[3], r[4]) >= 0.0
                          for pad_, li_ in swapped_pads)]
    return out, mouths


#: Exit corridors per (pad, grid step): the grid cells along each allowed
#: orthogonal exit, from the outline to two steps past its mouth.
_CORRIDOR_CACHE: Dict[tuple, list] = {}


def exit_corridor_cells(pad, width, step) -> List[Tuple[int, int]]:
    """The (gx, gy) cells a pad's straight exits run through outside its
    copper -- static geometry, no obstacle check: what reserve_exits keeps
    dear for the other nets."""
    key = (id(pad), round(step, 6), round(width, 6))
    hit = _CORRIDOR_CACHE.get(key)
    if hit is not None and hit[0] is pad:
        return hit[1]
    from pad_entry import smd_gradable, exit_distance
    cells = []
    layer = next((L for L in pad.layers if L.endswith('.Cu')), None)
    if layer and smd_gradable(pad, layer):
        run = exit_distance(width) + 2 * step
        for (ux, uy), (sx, sy), extent in _exit_dirs(pad):
            if sx and sy:
                continue
            t = extent
            while t <= extent + run + 1e-9:
                c = (int(round((pad.global_x + ux * t) / step)),
                     int(round((pad.global_y + uy * t) / step)))
                if not cells or cells[-1] != c:
                    cells.append(c)
                t += step / 2.0
    _CORRIDOR_CACHE[key] = (pad, cells)
    return cells


def reserve_exits(obstacles, pcb_data, net_ids, config) -> int:
    """Price the exit corridors (exit_corridor_cells) of every SMD pad of
    ``net_ids`` -- the nets still to route -- on ``obstacles``' stub
    proximity map at config.smd_exit_reserve_cost per cell, so a net routed
    now goes round the straight exits a later net will need instead of
    across them (an exit taken that way leaves its pad no compliant exit,
    and the route a plain pad landing). Returns the cells stamped."""
    cost_mm = getattr(config, 'smd_exit_reserve_cost', 0.0) or 0.0
    if (cost_mm <= 0 or not net_ids or obstacles is None
            or not getattr(config, 'smd_exit_terminals', True)):
        return 0
    skip = skip_net_ids(pcb_data)
    step = config.grid_step
    width = config.track_width
    pts = []
    for nid in net_ids:
        if nid in skip:
            continue
        for p in pcb_data.pads_by_net.get(nid, []) or ():
            pts.extend(exit_corridor_cells(p, width, step))
    if not pts:
        return 0
    obstacles.add_stub_proximity_costs_batch(pts, 1, config.cell_cost(cost_mm))
    return len(pts)


def strict_terminal_cells(cells, mouths, anchor, fences, pcb_data, net_id,
                          config, obstacles, layer_names, coord):
    """Strict exits on one end of a search whose terminals are a copper
    ISLAND (multipoint launch / landing cells): a cell on, or within its exit
    run of, a gradable pad of the net is no terminal -- a track launched there
    would leave that pad at an angle, through a corner or bend short of its
    exit -- so it is dropped, and each pad whose copper held a dropped cell
    is offered through its own exit mouths instead. ``mouths`` / ``anchor`` /
    ``fences`` (the _phase_pad_mouths dicts and list) are extended in place.
    Returns the new cell list; ``cells`` unchanged when exits are not strict."""
    if not _STRICT[0] or not cells or pcb_data is None:
        return cells
    from pad_entry import pad_depth, exit_distance
    pads = [p for p in (pcb_data.pads_by_net.get(net_id, []) or ())
            if getattr(p, 'size_x', 0) and getattr(p, 'size_y', 0)]
    if not pads:
        return cells
    keep, hit = [], {}
    for c in cells:
        if tuple(c[:3]) in mouths:
            keep.append(c)
            continue
        li = c[2]
        layer = layer_names[li] if li < len(layer_names) else None
        if layer is None:
            keep.append(c)
            continue
        x, y = coord.to_float(c[0], c[1])
        w = config.get_net_track_width(net_id, layer)
        reach = exit_distance(w) + w / 2.0
        drop = False
        for p in pads:
            if abs(p.global_x - x) > p.size_x + reach + 1 or abs(p.global_y - y) > p.size_y + reach + 1:
                continue
            if not exit_gradable(p, layer):
                continue
            d = pad_depth(p, x, y)
            if d > -reach:
                drop = True
                if d >= 0.0:
                    hit[(id(p), li)] = (p, li, layer, w)
        if not drop:
            keep.append(c)
    for (_k, (p, li, layer, w)) in hit.items():
        ms = exit_mouths(p, layer, li, w, config.grid_step, pcb_data, net_id,
                         config, obstacles)
        if not ms:
            STATS['no_exit'] += 1
            continue
        STATS['pads'] += 1
        fences.append((obstacles, p, li, config.track_width, config.grid_step))
        for gx, gy, l2, ax, ay, sx, sy, g in ms:
            if (gx, gy, l2) not in mouths:
                keep.append((gx, gy, l2))
                mouths[(gx, gy, l2)] = (sx, sy, g)
                anchor[(gx, gy, l2)] = (ax, ay)
    return keep


def net_pad_fences(pcb_data, net_id, obstacles, config, layer_names):
    """Strict exits: fence_pad arguments for EVERY gradable pad of the net on
    every routing layer it is on -- a route may touch its own net's pads only
    through their exit mouths, never pass over or graze one ([] when exits
    are not strict)."""
    if not _STRICT[0] or pcb_data is None or obstacles is None:
        return []
    out = []
    for p in pcb_data.pads_by_net.get(net_id, []) or ():
        for li, L in enumerate(layer_names):
            if exit_gradable(p, L):
                out.append((obstacles, p, li, config.track_width, config.grid_step))
    return out
