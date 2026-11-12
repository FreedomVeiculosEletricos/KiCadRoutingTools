#!/usr/bin/env python3
"""SMD pad exits (pad_entry.py), Altium's SMT rules: a track leaves an SMD pad
straight out through an edge -- not through a corner, not at an angle (SMD
Entry) -- and keeps straight for exit_distance before its first bend (SMD To
Corner). check_weird reports a failing exit; rebuild_pad_exits rebuilds it
from the pad centre the way Altium's router lays one.

Plain python3-runnable, no pytest."""
import math
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'py_tools'))

from kicad_parser import parse_kicad_pcb  # noqa: E402
from pcb_modification import rebuild_pad_exits, smd_exit_violations  # noqa: E402
from check_weird import check_weird  # noqa: E402
from check_connected import check_net_connectivity  # noqa: E402
import pad_entry  # noqa: E402

W = 0.3
# Net 1: SMD pad A 1.0 x 1.6 at (10,10) (x 9.5..10.5, y 9.2..10.8) and SMD
# pad B at (20,10). Net 3: a THT pad at (30,10). Net 4: SMD round pad at (40,10).
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
	(net 3 "N3")
	(net 4 "N4")
	(footprint "test:R"
		(layer "F.Cu")
		(uuid "aaaaaaaa-0000-0000-0000-000000000001")
		(at 10 10)
		(pad "1" smd rect (at 0 0) (size 1.0 1.6)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a1"))
		(pad "2" smd rect (at 10 0) (size 1.0 1.6)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a2"))
		(pad "3" thru_hole circle (at 20 0) (size 1.6 1.6) (drill 0.8)
			(layers "*.Cu" "*.Mask")
			(net 3 "N3")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a3"))
		(pad "4" thru_hole circle (at 20 4) (size 1.6 1.6) (drill 0.8)
			(layers "*.Cu" "*.Mask")
			(net 3 "N3")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a4"))
		(pad "5" smd circle (at 30 0) (size 1.2 1.2)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 4 "N4")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a5"))
		(pad "6" smd circle (at 30 4) (size 1.2 1.2)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 4 "N4")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a6"))
	)
"""
_SEG = """	(segment (start {x1} {y1}) (end {x2} {y2}) (width {w}) (layer "F.Cu")
		(net {net}) (uuid "bbbbbbbb-0000-0000-0000-0000000{i:05d}"))
"""


def _board(polylines):
    """polylines: [(points, width, net)]"""
    body, i = "", 0
    for pts, w, n in polylines:
        for a, b in zip(pts, pts[1:]):
            body += _SEG.format(x1=a[0], y1=a[1], x2=b[0], y2=b[1], w=w, net=n, i=i)
            i += 1
    d = tempfile.mkdtemp(prefix="smdexit_")
    p = os.path.join(d, "b.kicad_pcb")
    with open(p, "w") as f:
        f.write(_HEAD + body + ")\n")
    return parse_kicad_pcb(p)


def _cfg():
    return SimpleNamespace(clearance=0.15, track_width=W, net_clearances={},
                           board_edge_clearance=0.0, hole_to_hole_clearance=0.2,
                           grid_step=0.1)


def _rules(pcb):
    f, _ = check_weird(pcb, tolerance=0)
    return sorted({x['category'] for x in f if x['category'] in
                   ('smd-entry', 'smd-to-corner', 'shallow-pad-entry',
                    'narrow-pad-joint')})


def _connected(pcb, net=1):
    c = check_net_connectivity(net, [s for s in pcb.segments if s.net_id == net],
                               pcb.vias, pcb.pads_by_net[net], [])
    return c['num_components'] == 1 and not c['disconnected_pads']


# Right pad B (20,10): every fixture's far end is its centre, a clean exit
B_EXIT = [(20.0, 10.0), (18.5, 10.0)]


def test_straight_exit_is_clean():
    """Out of A's right end, straight 1 mm, then bend: compliant."""
    pcb = _board([([(10.0, 10.0), (11.5, 10.0), (12.5, 11.0), (17.5, 11.0),
                    (18.5, 10.0)] + B_EXIT[::-1][1:], W, 1)])
    assert _rules(pcb) == [], _rules(pcb)
    print("PASS straight perpendicular exit -> clean")


def test_corner_exit_is_flagged_and_rebuilt():
    """Diagonal out through A's top-right corner (the screenshots' case)."""
    pcb = _board([([(10.3, 10.6), (11.3, 11.6), (17.0, 11.6), (18.5, 10.1),
                    (20.0, 10.1)], W, 1)])
    assert 'smd-entry' in _rules(pcb), _rules(pcb)
    results = []
    n, strip, added = rebuild_pad_exits(results, pcb, {1}, _cfg())
    assert n >= 1, n
    assert _rules(pcb) == [], _rules(pcb)
    assert _connected(pcb)
    a = added[0]
    assert (a.start_x, a.start_y) == (10.0, 10.0), "rebuilt exit starts at the centre"
    print("PASS corner exit -> smd-entry, rebuilt from the pad centre")


def test_bend_on_the_edge_is_flagged_and_rebuilt():
    """Perpendicular out of the right end, but bent 0.05 mm past the edge."""
    pcb = _board([([(10.0, 10.0), (10.55, 10.0), (11.55, 11.0), (17.5, 11.0),
                    (18.5, 10.0), (20.0, 10.0)], W, 1)])
    assert _rules(pcb) == ['smd-to-corner'], _rules(pcb)
    n, _s, _a = rebuild_pad_exits([], pcb, {1}, _cfg())
    assert n == 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    print("PASS bend at the edge -> smd-to-corner, rebuilt")


def test_shallow_end_is_rebuilt_not_left_diagonal():
    """An end grazing A's corner from outside: rebuilt straight out."""
    pcb = _board([([(10.58, 10.86), (11.62, 11.9), (17.0, 11.9), (18.5, 10.4),
                    (18.5, 10.0), (20.0, 10.0)], W, 1)])
    assert {'shallow-pad-entry', 'narrow-pad-joint'} & set(_rules(pcb)), _rules(pcb)
    n, _s, _a = rebuild_pad_exits([], pcb, {1}, _cfg())
    assert n >= 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    print("PASS shallow corner graze -> rebuilt straight out")


def test_blocked_exit_is_left_and_reported():
    """Foreign copper fences every edge of A: nothing is rebuilt, the
    violation stays reported."""
    fence = [([(9.0, 8.6), (11.0, 8.6), (11.0, 11.4), (9.0, 11.4), (9.0, 8.6)], 0.1, 2)]
    pcb = _board([([(10.3, 10.6), (10.75, 11.05)], W, 1)] + fence)
    before = len(pcb.segments)
    n, _s, _a = rebuild_pad_exits([], pcb, {1}, _cfg())
    assert n == 0 and len(pcb.segments) == before
    assert 'smd-entry' in _rules(pcb)
    print("PASS blocked exit -> left alone, still reported")


def test_side_rule():
    """A 1.0 x 2.4 pad's side is > 2x its end: a side exit needs
    SMD_ENTRY_SIDE (Altium's Side option)."""
    pad = SimpleNamespace(size_x=1.0, size_y=2.4, shape='rect', rect_rotation=0.0,
                          global_x=0.0, global_y=0.0)
    assert set(pad_entry.allowed_exits(pad, allow_side=False)) == {(0.0, 1.0), (0.0, -1.0)}
    assert len(pad_entry.allowed_exits(pad, allow_side=True)) == 4
    sq = SimpleNamespace(size_x=1.0, size_y=1.6, shape='rect', rect_rotation=0.0,
                         global_x=0.0, global_y=0.0)
    assert len(pad_entry.allowed_exits(sq, allow_side=False)) == 4   # side < 2x end
    print("PASS SMD Entry side rule")


def test_round_pad_radial_and_tht_ignored():
    """A round SMD pad is left radially; a through-hole pad is not graded."""
    radial = _board([([(40.0, 10.0), (40.0, 11.0), (40.6, 11.6), (40.6, 12.4),
                       (40.0, 13.0), (40.0, 14.0)], W, 4)])
    assert _rules(radial) == [], _rules(radial)
    off = _board([([(40.2, 10.0), (40.2, 11.0), (40.8, 11.6), (40.8, 12.2),
                    (40.0, 13.0), (40.0, 14.0)], W, 4)])
    assert _rules(off) == ['smd-entry'], _rules(off)
    tht = _board([([(30.0, 10.0), (30.6, 10.6), (30.6, 14.0), (30.0, 14.0)], W, 3)])
    assert smd_exit_violations([s for s in tht.segments], [], tht.pads_by_net[3]) == []
    print("PASS round SMD pad radial, THT pad not graded")


TESTS = [test_straight_exit_is_clean, test_corner_exit_is_flagged_and_rebuilt,
         test_bend_on_the_edge_is_flagged_and_rebuilt,
         test_shallow_end_is_rebuilt_not_left_diagonal,
         test_blocked_exit_is_left_and_reported, test_side_rule,
         test_round_pad_radial_and_tht_ignored]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(TESTS)} passed")
