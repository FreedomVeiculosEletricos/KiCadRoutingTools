#!/usr/bin/env python3
"""SMD pad exit funnels (pad_entry.exit_funnel): the router reaches a terminal
SMD pad only straight out through an allowed edge, and keeps straight to the
SMD To Corner distance before it may turn -- Altium's SMD Entry and SMD To
Corner laid by the search, not rebuilt afterwards. The octolinear smoother
keeps such an exit (it may not collapse it into a diagonal off the pad).

Plain python3-runnable, no pytest."""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'rust_router'))

import numpy as np  # noqa: E402
import grid_router  # noqa: E402
from kicad_parser import Pad, Segment, PCBData, Net, BoardInfo  # noqa: E402
from pad_entry import exit_funnel, exit_distance, grade_pad_exit  # noqa: E402
from pcb_modification import smooth_octolinear_chains  # noqa: E402

G = 0.1
W = 0.3
fails = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if not ok else ''))
    if not ok:
        fails.append(name)


def _pad(x, y, sx, sy, rot=0.0, shape='rect'):
    return SimpleNamespace(size_x=sx, size_y=sy, shape=shape, rect_rotation=rot,
                           global_x=x, global_y=y, layers=['F.Cu'], drill=0)


def _route(cells, src, dst):
    om = grid_router.GridObstacleMap(2)
    if cells:
        om.set_move_masks_batch(np.array(
            [(gx, gy, li, m) for (gx, gy, li), m in cells.items()], dtype=np.int32))
    r = grid_router.GridRouter(via_cost=500000, h_weight=1.0)
    path, _it, _st = r.route_multi(om, [src], [dst], 200000, False, 0, None, None,
                                   0, 0, 0, 0.0, 0.0, 0, 0)
    return om, path


def _exit_pts(path, pad_xy):
    """The path from the pad end outward, in mm, the pad centre first."""
    pts = [(gx * G, gy * G) for gx, gy, _l in path]
    if (abs(pts[-1][0] - pad_xy[0]) < 1e-6 and abs(pts[-1][1] - pad_xy[1]) < 1e-6):
        pts = pts[::-1]
    return pts


def test_funnel_geometry():
    """A 1.0 x 1.6 pad: four exit lines from the centre, each straight to the
    exit distance past its edge; every other cell its copper would touch is
    closed."""
    pad = _pad(10.0, 10.0, 1.0, 1.6)
    cells, n = exit_funnel(pad, 0, W, G)
    check("four exits", n == 4, n)
    run = exit_distance(W)
    # right exit: x-edge at 10.5; cells along y=10.0 until x - 10.5 >= run
    right = sorted(gx for (gx, gy, _l), m in cells.items() if gy == 100 and gx > 100 and m)
    check("right line ends before the mouth",
          right and (right[-1] * G - 10.5) < run - 1e-9 and ((right[-1] + 1) * G - 10.5) >= run - 1e-9,
          right)
    check("line cells allow only their axis", cells[(102, 100, 0)] == (1 << 0) | (1 << 4))
    check("centre allows all four", cells[(100, 100, 0)] == 0xFF & ((1 << 0) | (1 << 2) | (1 << 4) | (1 << 6)))
    check("a corner cell is closed", cells.get((105, 108, 0)) == 0, cells.get((105, 108, 0)))
    check("far cells are free", (110, 110, 0) not in cells)


def test_off_grid_and_blocked_and_rotation():
    """An exit whose line is off the grid, or blocked to its mouth, is left
    out; a pad at 45 degrees gets no funnel."""
    pad = _pad(10.0, 10.05, 1.0, 1.6)          # y off the 0.1 grid
    cells, n = exit_funnel(pad, 0, W, G)
    check("an off-grid axis runs on the nearest grid line", n == 4, n)
    # fine pitch, x half a step off: the end exits run on the grid line
    # 0.05 mm beside the axis while a 0.2 mm track still fits the 0.3 mm edge
    vert = (1 << 2) | (1 << 6)
    cells, n = exit_funnel(_pad(10.05, 10.0, 0.3, 1.2), 0, 0.2, G)
    centre = next(m for (gx, gy, _l), m in cells.items() if gy == 100 and m & vert)
    check("a 0.3 mm pad half a step off keeps its end exits", n == 4 and centre & vert == vert, n)
    cells, n = exit_funnel(_pad(10.05, 10.0, 0.22, 1.2), 0, 0.2, G)
    check("a 0.22 mm pad half a step off loses them", n == 2
          and not any(m & vert for m in cells.values()), n)
    cells, why = exit_funnel(_pad(10.05, 10.05, 0.22, 0.22), 0, 0.2, G)
    check("a 0.22 mm square pad off both axes gets no funnel",
          cells is None and why == 'off-grid', why)
    pad = _pad(10.0, 10.0, 1.0, 1.6)
    cells, n = exit_funnel(pad, 0, W, G, blocked=lambda gx, gy: gx >= 104)
    check("blocked right exit dropped", n == 3, n)
    cells, why = exit_funnel(pad, 0, W, G, blocked=lambda gx, gy: True)
    check("all blocked -> no funnel", cells is None and why == 'blocked', why)
    # the geometry is cached per pad; which exits are blocked is not
    cells, n = exit_funnel(pad, 0, W, G)
    check("the same pad unblocked again gets all four exits", n == 4, n)
    cells, n = exit_funnel(pad, 1, W, G)
    check("...and its funnel on another layer is keyed to that layer",
          n == 4 and all(k[2] == 1 for k in cells), n)
    cells, why = exit_funnel(_pad(10.0, 10.0, 1.0, 1.6, rot=45.0), 0, W, G)
    check("45 degrees -> no funnel", cells is None and why == 'rotation', why)
    cells, why = exit_funnel(_pad(10.0, 10.0, 1.0, 1.0, shape='circle'), 0, W, G)
    check("round pad -> no funnel", cells is None and why == 'round', why)


def test_search_lays_compliant_exit():
    """Without the funnel the A* leaves the pad on the diagonal to a target
    up and to the right (SMD Entry fails); with it, straight out and straight
    to the exit distance."""
    pad = _pad(10.0, 10.0, 0.55, 0.5)
    tgt = (130, 70, 0)
    _om, raw = _route({}, (100, 100, 0), tgt)
    rules = grade_pad_exit(pad, _exit_pts(raw, (10.0, 10.0)), W)
    check("unconstrained search fails SMD Entry", 'smd-entry' in rules, rules)
    cells, _n = exit_funnel(pad, 0, W, G)
    om, path = _route(cells, (100, 100, 0), tgt)
    check("funnelled search finds a path", path is not None)
    rules = grade_pad_exit(pad, _exit_pts(path, (10.0, 10.0)), W)
    check("funnelled exit meets SMD Entry and SMD To Corner", rules == set(), rules)
    om.clear_move_masks()
    check("masks clear", om.move_mask_count() == 0)
    # the same pad as the TARGET: the funnel works both ways
    om, path = _route(cells, tgt, (100, 100, 0))
    rules = grade_pad_exit(pad, _exit_pts(path, (10.0, 10.0)), W)
    check("funnelled arrival meets both rules", rules == set(), rules)


def _board(drill):
    pads = [Pad(component_ref='U1', pad_number='1', global_x=10.0, global_y=10.0,
                local_x=0.0, local_y=0.0, size_x=0.5, size_y=0.5, shape='rect',
                layers=['F.Cu'], net_id=1, net_name='/SIG', drill=drill),
            Pad(component_ref='U2', pad_number='1', global_x=13.0, global_y=11.0,
                local_x=0.0, local_y=0.0, size_x=0.5, size_y=0.5, shape='rect',
                layers=['F.Cu'], net_id=1, net_name='/SIG', drill=drill)]
    pts = [(10.0, 10.0), (11.0, 10.0), (12.0, 11.0), (13.0, 11.0)]   # H D H
    segs = [Segment(start_x=a[0], start_y=a[1], end_x=b[0], end_y=b[1], width=W,
                    layer='F.Cu', net_id=1) for a, b in zip(pts, pts[1:])]
    return PCBData(footprints={}, nets={1: Net(1, '/SIG')}, segments=segs, vias=[],
                   board_info=BoardInfo(layers={}, copper_layers=['F.Cu', 'B.Cu'],
                                        board_bounds=(0.0, 0.0, 50.0, 50.0)),
                   pads_by_net={1: pads})


def test_smoother_keeps_smd_exit():
    """H-D-H between two SMD pads: the equal-length D-H collapse would leave
    a pad on the diagonal, so the smoother keeps the straight exits; between
    through-hole pads it still collapses."""
    pcb = _board(0)
    smooth_octolinear_chains([], pcb, clearance=0.1)
    pad = pcb.pads_by_net[1][0]
    chain = sorted(pcb.segments, key=lambda s: min(s.start_x, s.end_x))
    pts = [(chain[0].start_x, chain[0].start_y)] + [(s.end_x, s.end_y) for s in chain]
    if abs(pts[0][0] - 10.0) > 1e-6:
        pts = [(s.end_x, s.end_y) for s in chain[:1]] + [(s.start_x, s.start_y) for s in chain]
    check("SMD ends keep their straight exits", len(pcb.segments) == 3
          and grade_pad_exit(pad, pts, W) == set(), (len(pcb.segments), pts))
    pcb = _board(0.3)
    smooth_octolinear_chains([], pcb, clearance=0.1)
    check("through-hole ends still collapse", len(pcb.segments) == 2, len(pcb.segments))


def _route_pen(cells, src, dst, penalty, walls=()):
    om = grid_router.GridObstacleMap(2)
    om.set_move_masks_batch(np.array(
        [(gx, gy, li, m) for (gx, gy, li), m in cells.items()], dtype=np.int32))
    om.set_move_mask_penalty(penalty)
    for gx, gy, li in walls:
        om.add_blocked_cell(gx, gy, li)
    r = grid_router.GridRouter(via_cost=500000, h_weight=1.0)
    path, _it, _st = r.route_multi(om, [src], [dst], 400000, False, 0, None, None,
                                   0, 0, 0, 0.0, 0.0, 0, 0)
    return path


def test_soft_funnel_and_via_ban():
    """With every funnel line walled off, a hard funnel leaves no way out of
    the pad but a soft one lets the route through at its penalty; no via goes
    inside a funnel."""
    pad = _pad(10.0, 10.0, 0.55, 0.5)
    cells, _n = exit_funnel(pad, 0, W, G)
    lines = [k for k, m in cells.items() if m and k != (100, 100, 0)]
    walls = [(gx, gy, 0) for gx, gy, _l in lines]
    tgt = (130, 70, 0)
    check("hard funnel, lines walled: no path",
          _route_pen(cells, (100, 100, 0), tgt, 0, walls) is None)
    path = _route_pen(cells, (100, 100, 0), tgt, 20000, walls)
    check("soft funnel, lines walled: routes through at the penalty",
          path is not None and all(p[2] == 0 for p in path), path and path[:4])
    # the far target is on layer 1: the via must come after the funnel
    path = _route_pen(cells, (100, 100, 0), (130, 70, 1), 0)
    vias = [(a[0], a[1]) for a, b in zip(path, path[1:]) if a[2] != b[2]]
    check("no via inside the funnel", vias and all(
        (gx, gy, 0) not in cells for gx, gy in vias), vias)


def test_cleanup_guard():
    """worsens_smd_exits: a change that turns a straight exit diagonal is
    refused, one away from the pad or keeping the exit is not."""
    from pcb_modification import worsens_smd_exits
    pad = Pad(component_ref='U1', pad_number='1', global_x=10.0, global_y=10.0,
              local_x=0.0, local_y=0.0, size_x=0.5, size_y=0.5, shape='rect',
              layers=['F.Cu'], net_id=1, net_name='/SIG', drill=0)

    def segs(pts):
        return [Segment(start_x=a[0], start_y=a[1], end_x=b[0], end_y=b[1], width=W,
                        layer='F.Cu', net_id=1) for a, b in zip(pts, pts[1:])]
    straight = segs([(10.0, 10.0), (11.0, 10.0), (12.0, 11.0), (14.0, 11.0)])
    diagonal = segs([(10.0, 10.0), (11.0, 11.0), (14.0, 11.0)])
    check("a diagonal off the pad is refused",
          worsens_smd_exits(straight, diagonal, [], [pad]))
    far = straight[:2] + segs([(12.0, 11.0), (13.0, 11.5), (14.0, 11.0)])
    check("a change far from the pad is not", not worsens_smd_exits(straight, far, [], [pad]))
    check("an exit already failing may stay so",
          not worsens_smd_exits(diagonal, list(diagonal), [], [pad]))


def test_plane_tap_funnel():
    """A plane tap from a via up and to the right of an SMD pad leaves the pad
    straight out through an edge (route_planes._tap_search)."""
    from routing_config import GridRouteConfig
    from route_planes import route_via_to_pad
    pad = Pad(component_ref='C1', pad_number='1', global_x=10.0, global_y=10.0,
              local_x=0.0, local_y=0.0, size_x=0.6, size_y=0.5, shape='rect',
              layers=['F.Cu'], net_id=1, net_name='/GND', drill=0)
    cfg = GridRouteConfig()
    cfg.track_width = 0.2
    om = grid_router.GridObstacleMap(1)
    segs = route_via_to_pad((11.5, 8.7), pad, 'F.Cu', 1, om, cfg)
    pts = [(segs[-1]['end'][0], segs[-1]['end'][1])]
    for sg in reversed(segs):
        pts.append(sg['start'])
    check("the tap leaves the pad straight", grade_pad_exit(pad, pts, 0.2) == set(),
          (grade_pad_exit(pad, pts, 0.2), pts))
    check("the funnel is cleared after the tap", om.move_mask_count() == 0)
    cfg.smd_exit_funnel = False
    segs = route_via_to_pad((11.5, 8.7), pad, 'F.Cu', 1, om, cfg)
    pts = [(segs[-1]['end'][0], segs[-1]['end'][1])] + [sg['start'] for sg in reversed(segs)]
    check("without it the tap leaves at an angle", 'smd-entry' in grade_pad_exit(pad, pts, 0.2),
          pts)


if __name__ == '__main__':
    for fn in (test_funnel_geometry, test_off_grid_and_blocked_and_rotation,
               test_search_lays_compliant_exit, test_smoother_keeps_smd_exit,
               test_soft_funnel_and_via_ban, test_cleanup_guard,
               test_plane_tap_funnel):
        print(fn.__name__)
        fn()
    if fails:
        print(f"\nFAILED ({len(fails)}): {fails}")
        sys.exit(1)
    print("\nALL PASS")
