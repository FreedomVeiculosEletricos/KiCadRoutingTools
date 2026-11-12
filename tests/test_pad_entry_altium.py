#!/usr/bin/env python3
"""Pad entry (pad_entry.py): a track end that touches its pad without its own
width plus PAD_ENTRY_MARGIN_MM inside it is SHALLOW -- connected and DRC-clean,
but joined through less copper than the track. close_soft_joints lays a
connector to the pad centre (Altium's pad exit, Eagle's connection point),
check_weird reports what is left, and the strict collapse may not remove the
connector.

Plain python3-runnable, no pytest."""
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'py_tools'))

from kicad_parser import parse_kicad_pcb  # noqa: E402
from pcb_modification import close_soft_joints, collapse_strict_redundant  # noqa: E402
from check_weird import check_weird  # noqa: E402
import pad_entry  # noqa: E402

W = 0.3   # track width
# Net 1: pad A 1.0 x 1.0 at (10,10) (spans 9.5..10.5), pad B 1.0 x 1.0 at
# (10,14). Net 2: pad C, 0.3 x 1.2, at (16,10) (narrower than a 0.5 track).
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
	(footprint "test:R"
		(layer "F.Cu")
		(uuid "aaaaaaaa-0000-0000-0000-000000000001")
		(at 10 10)
		(pad "1" smd rect (at 0 0) (size 1.0 1.0)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a1"))
		(pad "2" smd rect (at 0 4) (size 1.0 1.0)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a2"))
	)
	(footprint "test:U"
		(layer "F.Cu")
		(uuid "aaaaaaaa-0000-0000-0000-000000000002")
		(at 16 10)
		(pad "1" smd rect (at 0 0) (size 0.3 1.2)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 2 "N2")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000b1"))
		(pad "2" smd rect (at 0 4) (size 0.3 1.2)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 2 "N2")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000b2"))
	)
"""
_SEG = """	(segment (start {x1} {y1}) (end {x2} {y2}) (width {w}) (layer "F.Cu")
		(net {net}) (uuid "bbbbbbbb-0000-0000-0000-0000000{i:05d}"))
"""


def _board(segments):
    body = "".join(_SEG.format(x1=a[0], y1=a[1], x2=b[0], y2=b[1], w=w, net=n, i=i)
                   for i, (a, b, w, n) in enumerate(segments))
    d = tempfile.mkdtemp(prefix="padentry_")
    p = os.path.join(d, "b.kicad_pcb")
    with open(p, "w") as f:
        f.write(_HEAD + body + ")\n")
    return parse_kicad_pcb(p)


def _cfg():
    return SimpleNamespace(clearance=0.15, track_width=W,
                           hole_to_hole_clearance=0.2, grid_step=0.1)


def _pad(pcb, ref, num):
    x, y = {('R', '1'): (10, 10), ('R', '2'): (10, 14),
            ('U', '1'): (16, 10), ('U', '2'): (16, 14)}[(ref, num)]
    return next(p for p in pcb.pads_by_net[1 if ref == 'R' else 2]
                if abs(p.global_x - x) < 1e-6 and abs(p.global_y - y) < 1e-6)


def _shallow(pcb):
    f, _ = check_weird(pcb, tolerance=0)
    return [x for x in f if x['category'] in ('shallow-pad-entry', 'narrow-pad-joint')]


def test_depth_geometry():
    pcb = _board([])
    a = _pad(pcb, 'R', '1')
    assert abs(pad_entry.pad_depth(a, 10.0, 10.0) - 0.5) < 1e-9
    assert abs(pad_entry.pad_depth(a, 10.0, 10.6) + 0.1) < 1e-9
    assert abs(pad_entry.segment_pad_depth(a, 10.0, 11.0, 10.0, 9.0) - 0.5) < 1e-6
    assert abs(pad_entry.required_depth(a, W) - (W / 2 + pad_entry.PAD_ENTRY_MARGIN_MM)) < 1e-9
    c = _pad(pcb, 'U', '1')
    assert abs(pad_entry.required_depth(c, 0.5) - 0.15) < 1e-9   # narrow: medial axis
    print("PASS depth geometry")


def test_edge_end_gets_centre_connector():
    """End 0.02 mm outside the edge, perpendicular: connected and full-width
    at the outline, but with no copper margin -> connector to the centre."""
    E = (10.0, 10.52)
    pcb = _board([((10.0, 14.0), E, W, 1)])
    assert _shallow(pcb), "fixture must be shallow"
    n = close_soft_joints([], pcb, {1}, _cfg())
    assert n == 1, n
    c = pcb.segments[-1]
    assert (c.start_x, c.start_y) == E and (c.end_x, c.end_y) == (10.0, 10.0), c
    assert abs(c.width - W) < 1e-9
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS edge end -> connector to the pad centre")


def test_corner_graze_gets_centre_connector():
    E = (9.42, 10.56)
    pcb = _board([((8.0, 12.0), E, W, 1)])
    n = close_soft_joints([], pcb, {1}, _cfg())
    assert n == 1 and (pcb.segments[-1].end_x, pcb.segments[-1].end_y) == (10.0, 10.0)
    assert not _shallow(pcb)
    print("PASS corner graze -> connector to the pad centre")


def test_deep_landing_untouched():
    """width/2 + margin = 0.2 mm inside: compliant, no copper added."""
    pcb = _board([((10.0, 14.0), (10.0, 10.3), W, 1)])
    assert not _shallow(pcb)
    assert close_soft_joints([], pcb, {1}, _cfg()) == 0
    print("PASS deep landing -> untouched")


def test_narrow_pad_needs_the_medial_axis():
    """A 0.5 mm track onto a 0.3 mm pin: on the pin's axis it is the
    deliberate neck-down (compliant); 0.1 mm off the axis it is shallow."""
    pcb = _board([((16.0, 7.0), (16.0, 9.8), 0.5, 2)])
    assert not _shallow(pcb)
    pcb = _board([((16.1, 7.0), (16.1, 9.8), 0.5, 2)])
    assert _shallow(pcb)
    assert close_soft_joints([], pcb, {2}, _cfg()) == 1
    assert not _shallow(pcb)
    print("PASS narrow pad -> medial axis required")


def test_blocked_centre_falls_back_to_shortest_connector():
    """Foreign copper over the pad centre's path: the connector stops where
    the required depth is met instead."""
    E = (10.0, 10.52)
    pcb = _board([((10.0, 14.0), E, W, 1),
                  ((9.0, 9.75), (11.0, 9.75), 0.1, 3)])   # foreign, across the pad
    n = close_soft_joints([], pcb, {1}, _cfg())
    assert n == 1, n
    c = pcb.segments[-1]
    need = pad_entry.required_depth(_pad(pcb, 'R', '1'), W)
    assert c.end_y > 10.0 and abs((10.5 - c.end_y) - need) < 2e-3, c
    print("PASS blocked centre -> shortest compliant connector")


def test_collapse_keeps_the_connector():
    """The strict collapse may not remove the connector: that would recreate
    the shallow entry. The end lies just INSIDE pad B's outline, so the strict
    graph joins the track to the pad without the connector and, absent the
    pad-entry test in StrictRemovalModel, the connector is an in-pad wiggle
    the collapse deletes."""
    E = (10.0, 13.55)                         # 0.05 mm into pad B, needs 0.2
    pcb = _board([((10.0, 10.0), (10.0, 11.0), W, 1),
                  ((10.0, 11.0), E, W, 1)])
    results = []
    assert close_soft_joints(results, pcb, {1}, _cfg()) == 1
    conn = results[-1]['new_segments'][0]
    collapse_strict_redundant(results, pcb, {1})
    assert any(s is conn for s in pcb.segments), "connector was collapsed"
    assert not _shallow(pcb)
    print("PASS strict collapse keeps the pad-entry connector")


def test_end_on_another_track_is_not_a_pad_entry():
    """A stub ending on the body of a same-net track that carries on to the
    pad centre is a T into that copper, not an entry: no connector (it would
    close a loop) and no report."""
    pcb = _board([((10.0, 13.0), (10.0, 10.0), W, 1),     # deep, to the centre
                  ((10.0, 13.0), (10.0, 10.52), W, 1)])   # stub on its body
    assert not _shallow(pcb)
    assert close_soft_joints([], pcb, {1}, _cfg()) == 0
    print("PASS end on another track -> not a pad entry")


TESTS = [test_depth_geometry, test_end_on_another_track_is_not_a_pad_entry, test_edge_end_gets_centre_connector,
         test_corner_graze_gets_centre_connector, test_deep_landing_untouched,
         test_narrow_pad_needs_the_medial_axis,
         test_blocked_centre_falls_back_to_shortest_connector,
         test_collapse_keeps_the_connector]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(TESTS)} passed")
