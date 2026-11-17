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


def _board(polylines, vias=()):
    """polylines: [(points, width, net)]; vias: [(x, y, net)] (0.5/0.25,
    F.Cu-B.Cu)"""
    body, i = "", 0
    for k, (x, y, n) in enumerate(vias):
        body += (f'\t(via (at {x} {y}) (size 0.5) (drill 0.25) '
                 f'(layers "F.Cu" "B.Cu") (net {n}) '
                 f'(uuid "cccccccc-0000-0000-0000-0000000{k:05d}"))\n')
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


def test_off_axis_inside_the_pad_is_rebuilt():
    """The exit leaves A's right side straight, but 0.3 mm off the pad's
    axis, and reaches the centre by a 45-degree stub across the land: the
    end-of-run exit grade fails it (in-pad-off-axis) and the end-of-run
    rebuild lays it from the centre on the axis."""
    pts = [(10.0, 10.0), (10.3, 10.3), (11.5, 10.3), (12.5, 11.3), (17.0, 11.3),
           (18.5, 10.0), (20.0, 10.0)]
    pcb = _board([(pts, W, 1)])
    segs = [s for s in pcb.segments if s.net_id == 1]
    assert not smd_exit_violations(segs, [], pcb.pads_by_net[1]), \
        "in-run, the exit grade is SMD Entry / To Corner only"
    found = smd_exit_violations(segs, [], pcb.pads_by_net[1], in_pad=True)
    assert any('in-pad-off-axis' in it[6] for it in found), [it[6] for it in found]
    f, _ = check_weird(pcb, tolerance=0)
    assert 'in-pad-off-axis' in {x['category'] for x in f}
    assert rebuild_pad_exits([], pcb, {1}, _cfg())[0] == 0, "in-run: left as is"
    n, _s, _a = rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)
    assert n >= 1 and _connected(pcb), n
    f, _ = check_weird(pcb, tolerance=0)
    cats = {x['category'] for x in f}
    assert not cats & {'in-pad-off-axis', 'smd-entry', 'smd-to-corner'}, cats
    print("PASS off-axis copper inside the pad -> in-pad-off-axis, rebuilt on the axis")


def test_exit_with_fewest_bends_wins():
    """The run leaves A through its bottom-right corner into a channel just
    right of A (x = 10.75) that runs down to the far end: the bottom edge
    faces the far end, but the right edge reaches the channel with one bend
    and less copper. Every allowed edge is tried; the fewest bends win."""
    pcb = _board([([(10.3, 10.6), (10.75, 11.05), (10.75, 20.0), (20.0, 20.0),
                    (20.0, 10.0)], W, 1)])
    assert 'smd-entry' in _rules(pcb), _rules(pcb)
    # in-run: the first clear edge in facing order (the bottom), unchanged
    n, _s, added = rebuild_pad_exits([], pcb, {1}, _cfg())
    assert n >= 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    a = added[0]
    assert (a.end_x, a.end_y) == (10.0, 11.05), (a.end_x, a.end_y)
    # end of run: every edge tried, the right end (fewest bends) wins
    pcb = _board([([(10.3, 10.6), (10.75, 11.05), (10.75, 20.0), (20.0, 20.0),
                    (20.0, 10.0)], W, 1)])
    n, _s, added = rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)
    assert n >= 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    a = added[0]
    assert (a.start_x, a.start_y) == (10.0, 10.0), (a.start_x, a.start_y)
    assert abs(a.end_y - 10.0) < 1e-6 and a.end_x > 10.5, \
        f"exit through the right end, got ({a.end_x}, {a.end_y})"
    print("PASS end of run: the exit with the fewest bends wins over the facing edge")


def test_clean_exit_relaid_when_an_end_is_straighter():
    """A clean exit (out of A's bottom end, 0.25 mm straight) that then turns
    right and runs down a channel beside A: its side reaches the channel
    with one bend instead of two. In-run it stays; the end-of-run pass
    re-lays it out of the side, square with the channel (no slanted leg)."""
    pts = [(10.0, 10.0), (10.0, 11.05), (10.75, 11.05), (10.75, 20.0),
           (20.0, 20.0), (20.0, 10.0)]
    pcb = _board([(pts, W, 1)])
    assert _rules(pcb) == [], _rules(pcb)
    assert rebuild_pad_exits([], pcb, {1}, _cfg())[0] == 0, "in-run: left as is"
    n, _s, added = rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)
    assert n == 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    got = sorted((round(a.start_x, 4), round(a.start_y, 4), round(a.end_x, 4),
                  round(a.end_y, 4)) for a in added)
    assert got == [(10.0, 10.0, 10.75, 10.0), (10.75, 10.0, 10.75, 20.0)], got
    # a clean exit with nothing better is left alone
    pcb = _board([([(10.0, 10.0), (10.0, 12.0), (20.0, 12.0), (20.0, 10.0)], W, 1)])
    assert rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)[0] == 0
    print("PASS clean exit re-laid at the end of the run when a side is straighter")


def test_junction_inside_the_pad_is_graded_and_rebuilt():
    """Three tracks meet at A's centre (a stub, a run straight down and a
    run that leaves through A's top-right corner): the last is an exit too,
    though its end inside the pad is no free end -- graded only with
    ``through``, rebuilt at the end of the run."""
    pcb = _board([([(10.2, 10.0), (10.0, 10.0)], W, 1),
                  ([(10.0, 10.0), (10.0, 7.0)], W, 1),
                  ([(10.0, 10.0), (11.5, 11.5), (17.0, 11.5), (18.5, 10.0),
                    (20.0, 10.0)], W, 1)])
    segs = pcb.segments
    assert not smd_exit_violations(segs, [], pcb.pads_by_net[1])
    found = smd_exit_violations(segs, [], pcb.pads_by_net[1], through=True)
    assert [(e, r) for _s, e, _p, _pt, _c, _r, r in found] == \
        [((10.0, 10.0), {'smd-entry'})], found
    assert 'smd-entry' in _rules(pcb), _rules(pcb)
    n, _s, _a = rebuild_pad_exits([], pcb, {1}, _cfg())
    assert n == 0, "in-run: a junction is left as routed"
    n, _s, added = rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)
    assert n == 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    assert (added[0].start_x, added[0].start_y) == (10.0, 10.0)
    print("PASS junction inside the pad -> graded, rebuilt at the end")


def test_track_through_the_pad_is_split_and_rebuilt():
    """A run enters A slanted through its top edge, turns inside it and
    leaves straight down through its bottom edge, with no end in the pad:
    split at the point nearest the centre, the slanted half fails and is
    rebuilt from the centre; the straight half keeps its copper."""
    pcb = _board([(list(reversed(B_EXIT)) + [(18.5, 10.0), (17.0, 12.0),
                                             (12.0, 12.0), (10.0, 10.5),
                                             (10.0, 7.0)], W, 1)])
    segs = pcb.segments
    assert not smd_exit_violations(segs, [], pcb.pads_by_net[1])
    found = smd_exit_violations(segs, [], pcb.pads_by_net[1], through=True)
    got = sorted((e, sorted(r)) for _s, e, _p, _pt, _c, _r, r in found)
    assert got == [((10.0, 10.0), ['smd-entry'])], got
    n, _s, added = rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)
    assert n >= 1 and _rules(pcb) == [] and _connected(pcb), (n, _rules(pcb))
    keep = [a for a in added if (a.end_x, a.end_y) == (10.0, 7.0)]
    assert keep and (keep[0].start_x, keep[0].start_y) == (10.0, 10.0), \
        [(a.start_x, a.start_y, a.end_x, a.end_y) for a in added]
    print("PASS track through the pad -> split, slanted half rebuilt")


def test_track_grazing_the_side_is_flagged():
    """A run whose copper only touches A's top edge, its centreline
    outside the pad: an smd-entry, left as routed (no end to lay from)."""
    pcb = _board([(list(reversed(B_EXIT)) + [(17.5, 10.9), (9.0, 10.9)], W, 1)])
    found = smd_exit_violations(pcb.segments, [], pcb.pads_by_net[1],
                                through=True)
    assert [(r, rules) for _s, _e, p, _pt, _c, r, rules in found
            if p.pad_number == '1'] == [('graze', {'smd-entry'})], found
    assert 'smd-entry' in _rules(pcb), _rules(pcb)
    before = len(pcb.segments)
    stats = {}
    rebuild_pad_exits([], pcb, {1}, _cfg(), final=True, stats=stats)
    assert stats.get('grazes_pad') == 1, stats
    print("PASS track grazing the pad's side -> smd-entry, reported")


def test_parallel_runs_to_one_via_leave_one_exit():
    """A routed dogbone and a straight via->pad stub laid beside it both run
    from A's centre to one via (a loop): each blocks the other's rebuild
    with an acute corner at the via. The end-of-run pass rebuilds one and
    removes the other."""
    via = (11.0, 11.3)
    pcb = _board([([(10.0, 10.0), (10.2, 10.0), (10.2, 10.5), (11.0, 11.3)], W, 1),
                  ([(10.0, 10.0), (11.0, 11.3)], W, 1),
                  ([(11.0, 11.3), (11.0, 12.5), (18.5, 12.5), (20.0, 11.0),
                    (20.0, 10.0)], W, 1)],
                 vias=[(via[0], via[1], 1)])
    found = smd_exit_violations(pcb.segments, pcb.vias, pcb.pads_by_net[1],
                                through=True)
    assert len([f for f in found if f[2].pad_number == '1']) == 2, found
    n, _s, added = rebuild_pad_exits([], pcb, {1}, _cfg(), final=True)
    assert n >= 1 and _connected(pcb), n
    at_a = [s for s in pcb.segments
            if (s.start_x, s.start_y) == (10.0, 10.0)
            or (s.end_x, s.end_y) == (10.0, 10.0)]
    assert len(at_a) == 1, [(s.start_x, s.start_y, s.end_x, s.end_y) for s in at_a]
    assert not smd_exit_violations(pcb.segments, pcb.vias, pcb.pads_by_net[1],
                                   through=True)
    print("PASS parallel runs to one via -> one exit left")


TESTS = [test_straight_exit_is_clean, test_off_axis_inside_the_pad_is_rebuilt,
         test_corner_exit_is_flagged_and_rebuilt,
         test_bend_on_the_edge_is_flagged_and_rebuilt,
         test_shallow_end_is_rebuilt_not_left_diagonal,
         test_blocked_exit_is_left_and_reported, test_side_rule,
         test_round_pad_radial_and_tht_ignored,
         test_exit_with_fewest_bends_wins,
         test_clean_exit_relaid_when_an_end_is_straighter,
         test_junction_inside_the_pad_is_graded_and_rebuilt,
         test_track_through_the_pad_is_split_and_rebuilt,
         test_track_grazing_the_side_is_flagged,
         test_parallel_runs_to_one_via_leave_one_exit]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(TESTS)} passed")
