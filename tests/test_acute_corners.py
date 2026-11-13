#!/usr/bin/env python3
"""Corner angles: the sharpest corner the router lays and the acute corners
the cleanup cuts (corner_angle.py, Altium's Routing Corners / Acute Angle).

  - GridRouter ``max_turn_units`` is a hard limit on the corner a path makes
    over the last ``turn_window`` steps: a U-turn through a gap in a wall is
    a 180-degree corner when its legs are closer than the window, and with
    the limit at 90 degrees the router widens it instead. The control (free)
    takes the tight U.
  - ``acute_turn_cost`` prices a corner past 90 degrees: with it the router
    goes round where a free one cuts back sharply.
  - acute_corners grades end-to-end corners and T joints, not inside the
    net's own pad; fix_acute_corners cuts a corner octolinearly, keeps the
    net connected, and leaves one it cannot cut clear (reported).

Plain python3-runnable, no pytest.
"""
import math
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rust_router"))
sys.path.insert(0, os.path.join(ROOT, "py_router"))
import grid_router  # noqa: E402
from kicad_parser import parse_kicad_pcb  # noqa: E402
from corner_angle import acute_corners  # noqa: E402
from pcb_modification import fix_acute_corners  # noqa: E402
from check_connected import check_net_connectivity  # noqa: E402
from check_weird import check_weird  # noqa: E402
from routing_config import GridRouteConfig  # noqa: E402

OCT = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
       (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}


def _window_corner(path, window):
    """Largest corner (45-degree units) the path makes over ``window`` steps."""
    dirs = []
    for a, b in zip(path, path[1:]):
        if a[2] != b[2]:
            dirs.append(None)
            continue
        dirs.append(OCT[((b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1]))])
    worst = 0
    for i, d in enumerate(dirs):
        if d is None:
            continue
        for k in range(max(0, i - window), i):
            if any(x is None for x in dirs[k:i]):
                continue
            delta = (d - dirs[k]) % 8
            worst = max(worst, min(delta, 8 - delta))
    return worst


def _wall_map():
    """One layer; a wall along y = 12 with a 3-cell gap at x = 30..32."""
    obs = grid_router.GridObstacleMap(1)
    for x in range(-40, 80):
        if not 30 <= x <= 32:
            obs.add_blocked_cell(x, 12, 0)
    return obs


def _route(**kw):
    r = grid_router.GridRouter(via_cost=500, h_weight=1.0, turn_cost=1000, **kw)
    path, _it, _st = r.route_multi(_wall_map(), [(10, 10, 0)], [(10, 14, 0)],
                                   2_000_000, False, 0, None, None, 0, 0)
    return path


def test_max_turn_is_hard_over_the_window():
    free = _route(turn_window=8)
    assert free, "control: free route failed"
    assert _window_corner(free, 8) == 4, _window_corner(free, 8)
    held = _route(max_turn_units=2, turn_window=8)
    assert held, "90-degree limit: route failed (a wide U fits)"
    assert _window_corner(held, 8) <= 2, _window_corner(held, 8)
    assert len(held) > len(free)
    # the window is what makes the limit bite: per step the tight U is 90s
    tight = _route(max_turn_units=2, turn_window=1)
    assert tight and _window_corner(tight, 1) <= 2
    print("PASS max_turn_units: hard over the window; the free route takes the tight U")


def test_acute_cost_avoids_the_hairpin():
    free = _route(turn_window=8)
    priced = _route(acute_turn_cost=200_000, turn_window=8)
    assert priced
    assert _window_corner(priced, 8) <= 2 < _window_corner(free, 8), (
        _window_corner(priced, 8), _window_corner(free, 8))
    print("PASS acute_turn_cost: the priced route keeps its corners at 90 or wider")


def test_config_kwargs():
    c = GridRouteConfig(track_width=0.5, grid_step=0.1, max_turn_angle=90)
    k = c.corner_kwargs()
    assert k == {'max_turn_units': 2, 'acute_turn_cost': c.acute_turn_cost,
                 'turn_window': 5}, k
    assert c.min_interior_angle() == 90.0
    c = GridRouteConfig(max_turn_angle=45, min_corner_angle=90)
    assert c.corner_kwargs()['max_turn_units'] == 1
    assert c.min_interior_angle() == 135.0
    assert GridRouteConfig(min_corner_angle=0).min_interior_angle() == 0.0
    assert GridRouteConfig().corner_kwargs()['max_turn_units'] == 4   # free by default
    print("PASS config: angle -> units, window = track width in steps, min angle")


_HEAD = """(kicad_pcb
	(version 20260206)
	(generator "pcbnew")
	(generator_version "10.0")
	(general (thickness 1.6))
	(paper "A4")
	(layers
		(0 "F.Cu" signal)
		(2 "B.Cu" signal)
		(25 "Edge.Cuts" user)
	)
	(setup (pad_to_mask_clearance 0))
	(net 0 "")
	(net 1 "N1")
	(net 2 "N2")
	(footprint "test:J"
		(layer "F.Cu")
		(uuid "cccccccc-0000-0000-0000-000000000001")
		(at 10 10)
		(pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8)
			(layers "*.Cu" "*.Mask")
			(net 1 "N1")
			(uuid "cccccccc-0000-0000-0000-0000000000a1"))
		(pad "2" thru_hole circle (at 0 6) (size 1.6 1.6) (drill 0.8)
			(layers "*.Cu" "*.Mask")
			(net 1 "N1")
			(uuid "cccccccc-0000-0000-0000-0000000000a2"))
	)
"""
_SEG = """	(segment (start {x1} {y1}) (end {x2} {y2}) (width {w}) (layer "F.Cu")
		(net {net}) (uuid "dddddddd-0000-0000-0000-0000000{i:05d}"))
"""


def _board(polylines):
    body, i = "", 0
    for pts, w, n in polylines:
        for a, b in zip(pts, pts[1:]):
            body += _SEG.format(x1=a[0], y1=a[1], x2=b[0], y2=b[1], w=w, net=n, i=i)
            i += 1
    d = tempfile.mkdtemp(prefix="acute_")
    p = os.path.join(d, "b.kicad_pcb")
    with open(p, "w") as f:
        f.write(_HEAD + body + ")\n")
    return parse_kicad_pcb(p)


def _cfg():
    return SimpleNamespace(clearance=0.15, track_width=0.3, net_clearances={},
                           board_edge_clearance=0.0, hole_to_hole_clearance=0.2,
                           grid_step=0.1, min_interior_angle=lambda: 90.0)


def _connected(pcb, net=1):
    c = check_net_connectivity(net, [s for s in pcb.segments if s.net_id == net],
                               pcb.vias, pcb.pads_by_net[net], [])
    return c['num_components'] == 1 and not c['disconnected_pads']


def _net(pcb, n=1):
    return [s for s in pcb.segments if s.net_id == n]


# Pad 1 (10,10) -> out east to (14,10) -> back south-west to (12,12) -> down
# to pad 2 (10,16): the corner at (14,10) is 45 degrees (east then SW).
SPIKE = [(10.0, 10.0), (14.0, 10.0), (12.0, 12.0), (12.0, 14.0), (10.0, 16.0)]


def test_detect():
    pcb = _board([(SPIKE, 0.3, 1)])
    found = acute_corners(_net(pcb), pcb.vias, pcb.pads_by_net[1], 90)
    assert [(c.kind, round(c.x, 3), round(c.y, 3), round(c.angle)) for c in found] == \
        [('corner', 14.0, 10.0, 45)], found
    # a 90-degree corner passes a 90 rule, fails a 135 one (max turn 45)
    sq = _board([([(10.0, 10.0), (14.0, 10.0), (14.0, 13.0), (10.0, 16.0)], 0.3, 1)])
    assert not [c for c in acute_corners(_net(sq), [], sq.pads_by_net[1], 90)
                if abs(c.x - 14) < 1e-6 and abs(c.y - 10) < 1e-6]
    assert [c for c in acute_corners(_net(sq), [], sq.pads_by_net[1], 135)
            if abs(c.x - 14) < 1e-6 and abs(c.y - 10) < 1e-6]
    # a T at 45 degrees
    tee = _board([([(10.0, 10.0), (10.0, 16.0)], 0.3, 1),
                  ([(10.0, 13.0), (12.0, 11.0)], 0.3, 1)])
    kinds = [(c.kind, round(c.angle)) for c in acute_corners(_net(tee), [], tee.pads_by_net[1], 90)]
    assert kinds == [('tee', 45)], kinds
    # inside the net's own pad copper: not graded
    inpad = _board([([(14.0, 10.0), (10.0, 10.0), (10.6, 10.6), (14.0, 14.0)], 0.3, 1)])
    assert not acute_corners(_net(inpad), [], inpad.pads_by_net[1], 90)
    # check_weird reports it
    f, _ = check_weird(pcb, tolerance=0)
    assert [x for x in f if x['category'] == 'acute-corner'], f
    f, _ = check_weird(pcb, tolerance=0, min_corner_angle=0)
    assert not [x for x in f if x['category'] == 'acute-corner']
    print("PASS detection: end-to-end corner, 90 vs 135 rule, T joint, in-pad skipped, check_weird")


def test_fix_cuts_octolinearly():
    pcb = _board([(SPIKE, 0.3, 1)])
    results = []
    n, strip, added = fix_acute_corners(results, pcb, {1}, _cfg())
    assert n == 1, n
    assert not acute_corners(_net(pcb), pcb.vias, pcb.pads_by_net[1], 90)
    assert _connected(pcb)
    for s in _net(pcb):
        dx, dy = abs(s.end_x - s.start_x), abs(s.end_y - s.start_y)
        assert dx < 1e-6 or dy < 1e-6 or abs(dx - dy) < 1e-4, (s.start_x, s.start_y, s.end_x, s.end_y)
    assert len(strip) == 2, strip       # the two input tracks of the corner
    print("PASS fix: corner cut octolinearly, net still connected")


def test_fix_takes_the_whole_spike_and_leaves_a_joint():
    # foreign copper inside the spike: the largest clear cut drops the spike
    fence = ([(12.6, 10.4), (13.6, 10.4)], 0.2, 2)
    pcb = _board([(SPIKE, 0.3, 1), fence])
    n, _s, added = fix_acute_corners([], pcb, {1}, _cfg())
    assert n == 1 and _connected(pcb), n
    assert [(a.start_x, a.start_y, a.end_x, a.end_y) for a in added] == [(12.0, 12.0, 10.0, 10.0)], added
    print("PASS fix: the whole spike when it clears")


def _octolinear(pcb, net=1):
    for s in _net(pcb, net):
        dx, dy = abs(s.end_x - s.start_x), abs(s.end_y - s.start_y)
        if not (dx < 1e-6 or dy < 1e-6 or abs(dx - dy) < 1e-4):
            return False
    return True


def test_fix_y_tee_and_fold():
    # Y: a third track on the spike's vertex -> the branch moves along the
    # trunk and leaves it square
    pcb = _board([(SPIKE, 0.3, 1), ([(14.0, 10.0), (16.0, 10.0)], 0.3, 1)])
    stats = {}
    n, _s, _a = fix_acute_corners([], pcb, {1}, _cfg(), stats=stats)
    assert n >= 1 and not acute_corners(_net(pcb), [], pcb.pads_by_net[1], 90), (n, stats)
    assert _connected(pcb) and _octolinear(pcb)
    # T: a track ending on another's body at 45 degrees
    tee = _board([([(10.0, 10.0), (10.0, 16.0)], 0.3, 1),
                  ([(10.0, 13.0), (12.0, 11.0), (16.0, 11.0)], 0.3, 1)])
    n, _s, _a = fix_acute_corners([], tee, {1}, _cfg())
    assert n == 1 and not acute_corners(_net(tee), [], tee.pads_by_net[1], 90), n
    assert _connected(tee) and _octolinear(tee)
    # fold: out to (14,10) and back to (12,10) over itself, then down
    fold = _board([([(10.0, 10.0), (14.0, 10.0), (12.0, 10.0), (12.0, 13.0),
                     (10.0, 15.0), (10.0, 16.0)], 0.3, 1)])
    found = acute_corners(_net(fold), [], fold.pads_by_net[1], 90)
    assert [round(c.angle) for c in found] == [0], found
    n, _s, _a = fix_acute_corners([], fold, {1}, _cfg())
    assert n == 1 and not acute_corners(_net(fold), [], fold.pads_by_net[1], 90)
    assert _connected(fold)
    assert max(max(s.start_x, s.end_x) for s in _net(fold)) == 12.0   # the spike is gone
    print("PASS fix: Y joint and T squared off along the trunk, fold removed")


TESTS = [test_max_turn_is_hard_over_the_window, test_acute_cost_avoids_the_hairpin,
         test_config_kwargs, test_detect, test_fix_cuts_octolinearly,
         test_fix_takes_the_whole_spike_and_leaves_a_joint,
         test_fix_y_tee_and_fold]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(TESTS)} passed")
