#!/usr/bin/env python3
"""Pad entry (pad_entry.py): a track end that touches its pad without its own
width plus PAD_ENTRY_MARGIN_MM inside it is SHALLOW -- connected and DRC-clean,
but joined through less copper than the track. check_weird reports one, and
the passes that remove copper (the strict collapse, the dead-end sweep) refuse
to create one: they never trim a track short of its pad.

Plain python3-runnable, no pytest."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, os.path.join(ROOT, 'py_tools'))

from kicad_parser import parse_kicad_pcb  # noqa: E402
from pcb_modification import (collapse_strict_redundant, sweep_dead_ends,  # noqa: E402
                              prune_grazing_segments, nudge_grazing_microshift)
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


def test_edge_end_is_shallow():
    """End 0.02 mm outside the edge, perpendicular: connected and full-width
    at the outline, but with no copper margin -> reported."""
    pcb = _board([((10.0, 14.0), (10.0, 10.52), W, 1)])
    assert _shallow(pcb), "edge end must be reported"
    print("PASS edge end -> shallow-pad-entry")


def test_corner_graze_is_shallow():
    pcb = _board([((8.0, 12.0), (9.42, 10.56), W, 1)])
    assert _shallow(pcb)
    print("PASS corner graze -> reported")


def test_deep_landing_is_compliant():
    """width/2 + margin = 0.2 mm inside: compliant."""
    pcb = _board([((10.0, 14.0), (10.0, 10.3), W, 1)])
    assert not _shallow(pcb)
    print("PASS deep landing -> not reported")


def test_narrow_pad_needs_the_medial_axis():
    """A 0.5 mm track onto a 0.3 mm pin: on the pin's axis it is the
    deliberate neck-down (compliant); 0.1 mm off the axis it is shallow."""
    pcb = _board([((16.0, 7.0), (16.0, 9.8), 0.5, 2)])
    assert not _shallow(pcb)
    pcb = _board([((16.1, 7.0), (16.1, 9.8), 0.5, 2)])
    assert _shallow(pcb)
    print("PASS narrow pad -> medial axis required")


def test_end_on_another_track_is_not_a_pad_entry():
    """A stub ending on the body of a same-net track that carries on to the
    pad centre is a T into that copper, not an entry: not reported."""
    pcb = _board([((10.0, 13.0), (10.0, 10.0), W, 1),     # deep, to the centre
                  ((10.0, 13.0), (10.0, 10.52), W, 1)])   # stub on its body
    assert not _shallow(pcb)
    print("PASS end on another track -> not a pad entry")


def _into_pad_b(stop_y):
    """A route from pad A's centre down to pad B's centre, with a vertex at
    ``stop_y`` inside pad B (13.5..14.5). The strict graph joins the vertex to
    the pad, so the last leg is in-pad copper the collapse calls redundant."""
    return _board([((10.0, 10.0), (10.0, 11.0), W, 1),
                   ((10.0, 11.0), (10.0, stop_y), W, 1),
                   ((10.0, stop_y), (10.0, 14.0), W, 1)])


def _has_leg(pcb, stop_y):
    return any(abs(s.start_y - stop_y) < 1e-9 and abs(s.end_y - 14.0) < 1e-9
               for s in pcb.segments)


def test_collapse_does_not_trim_short_of_the_pad():
    """0.05 mm into pad B (needs 0.2): removing the leg to the centre would
    leave a shallow entry, so the strict collapse keeps it."""
    pcb = _into_pad_b(13.55)
    collapse_strict_redundant([], pcb, {1})
    assert _has_leg(pcb, 13.55), "the collapse trimmed the track short of its pad"
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS strict collapse keeps the leg into the pad")


def test_collapse_still_trims_a_deep_leg():
    """0.3 mm into pad B (needs 0.2): the end left is compliant, so the guard
    does not stop the collapse from removing the redundant in-pad leg."""
    pcb = _into_pad_b(13.8)
    collapse_strict_redundant([], pcb, {1})
    assert not _has_leg(pcb, 13.8), "a compliant trim was refused"
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS strict collapse trims a leg that leaves a deep end")


def test_collapse_drops_the_stub_not_the_entry():
    """A stub lying along the track that runs into pad B, ending at the pad
    edge: while that track is there the stub's end is a T into it, not a pad
    entry. Removing the track instead of the stub would expose the stub's end
    0.02 mm into the pad; the collapse must remove the stub."""
    pcb = _board([((10.0, 10.0), (10.0, 11.0), W, 1),
                  ((10.0, 11.0), (10.0, 14.0), W, 1),
                  ((10.0, 11.0), (10.0, 13.52), W, 1)])
    collapse_strict_redundant([], pcb, {1})
    assert not _shallow(pcb), _shallow(pcb)
    assert any(abs(s.end_y - 14.0) < 1e-9 for s in pcb.segments), \
        "the track into the pad centre was removed"
    print("PASS strict collapse drops the stub, keeps the entry")


def test_graze_prune_does_not_trim_short_of_the_pad():
    """A route arrives along pad B's lower edge and turns to its centre; a
    foreign track runs just below the pad, grazing the route AND the pad.
    The in-pad leg grazes it too, and the net stays connected without it --
    but dropping it clears nothing (the pad grazes the same track) and leaves
    the route's end 0.1 mm into the pad (needs 0.2): the prune keeps it."""
    leg = ((10.4, 14.35), (10.0, 14.0))
    pcb = _board([((10.0, 10.0), (10.0, 11.0), W, 1),
                  ((10.0, 11.0), (12.0, 11.0), W, 1),
                  ((12.0, 11.0), (12.0, 14.35), W, 1),
                  ((12.0, 14.35), leg[0], W, 1),
                  (leg[0], leg[1], W, 1),
                  ((9.0, 14.62), (12.5, 14.62), 0.1, 3)])
    prune_grazing_segments([], pcb, {1}, clearance=0.15,
                           check_foreign_segments=True)
    assert any((s.start_x, s.start_y, s.end_x, s.end_y)
               == (*leg[0], *leg[1]) for s in pcb.segments), \
        "the in-pad leg was pruned"
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS graze prune keeps the leg into the pad")


def test_sweep_does_not_trim_short_of_the_pad():
    """The dead-end sweep's soft-joint restore guard has the same twin. A
    route that crosses pad B and runs on to a dead end beyond it: removing the
    whole tail segment would leave an end 0.05 mm into the pad, so the guard
    puts it back."""
    pcb = _board([((10.0, 10.0), (10.0, 11.0), W, 1),
                  ((10.0, 11.0), (10.0, 13.55), W, 1),
                  ((10.0, 13.55), (10.0, 15.2), W, 1)])
    sweep_dead_ends([], pcb, {1})
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS dead-end sweep does not leave a shallow entry")


def test_restored_stub_into_the_pad_is_adopted():
    """#468: a terminal restore put a ripped net's escape stub back -- pad A's
    centre out to (10, 10.55). The net's new route ends on the stub's outer
    end, its cap already grazing pad A, so connectivity calls the stub
    redundant; dropping it leaves the route's end touching the pad from
    outside. The commit adopts it: it is what carries the end into the pad."""
    from pcb_modification import adopt_restored_stubs
    pcb = _board([((10.0, 14.0), (10.0, 10.55), W, 1),       # the new route
                  ((10.0, 10.55), (10.0, 10.0), W, 1)])      # the restored stub
    route, stub = pcb.segments
    pcb._stub_restores = {1: [stub]}
    result = {'new_segments': [route], 'new_vias': []}
    adopt_restored_stubs(pcb, result)
    assert stub in result['new_segments'], "the stub into the pad was dropped"
    pcb.segments = list(result['new_segments'])
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS restored stub into the pad -> adopted")


def test_microshift_does_not_slide_an_end_off_the_pad_axis():
    """A 0.3 mm track ends on the medial axis of the 0.3 mm pin C (the deepest
    the pin allows); a foreign track 0.02 mm inside clearance faces the end.
    Sliding the end 0.024 mm away clears the graze but leaves it off the axis
    -- joined through less than the track. The micro-shift refuses that move
    (the graze stays for the DRC report, as for any move it cannot make)."""
    pcb = _board([((16.0, 14.0), (16.0, 9.8), W, 2),
                  ((16.48, 9.8), (16.7, 9.8), 0.2, 3)])
    nudge_grazing_microshift([], pcb, {2}, clearance=0.25, max_shift=0.025)
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS micro-shift keeps the end on the pad axis")


_PAD_B = '''		(pad "2" smd rect (at 0 4) (size 1.0 1.0)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a2"))
'''
_PAD_C = '''		(pad "3" smd rect (at 0 8) (size 1.0 1.0)
			(layers "F.Cu" "F.Mask" "F.Paste")
			(net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000a3"))
'''


def _board3(segments):
    """_board with a third N1 pad C, 1.0 x 1.0, at (10,18)."""
    global _HEAD
    head = _HEAD
    assert _PAD_B in head
    _HEAD = head.replace(_PAD_B, _PAD_B + _PAD_C)
    try:
        return _board(segments)
    finally:
        _HEAD = head


def test_passing_track_is_shallow():
    """A track that passes along pad B's edge, 0.05 mm in, and carries on to
    pad C joins B with no end in it: reported. The same track run through
    B's centre is not."""
    pcb = _board3([((10.0, 10.0), (10.45, 11.0), W, 1),
                   ((10.45, 11.0), (10.45, 17.0), W, 1),
                   ((10.45, 17.0), (10.0, 18.0), W, 1)])
    f = _shallow(pcb)
    assert len(f) == 1 and '.2:' in f[0]['detail'], f
    pcb = _board3([((10.0, 10.0), (10.0, 18.0), W, 1)])
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS track passing the pad edge -> reported")


def _bend_on_pad_b(bend_y):
    """A -> B -> C with a bend at (10, bend_y) inside pad B, a leg from the
    bend to B's centre and the track carrying on from the bend to C. The leg
    is in-pad copper the strict graph calls redundant."""
    return _board3([((10.0, 10.0), (10.0, bend_y), W, 1),
                    ((10.0, bend_y), (10.0, 14.0), W, 1),
                    ((10.0, bend_y), (11.5, bend_y), W, 1),
                    ((11.5, bend_y), (11.5, 18.0), W, 1),
                    ((11.5, 18.0), (10.0, 18.0), W, 1)])


def test_collapse_does_not_leave_a_shallow_bend():
    """The bend is 0.05 mm into pad B (needs 0.2): dropping the leg to the
    centre leaves B joined only by the bend, with no end in it. The collapse
    keeps the leg; with the bend 0.3 mm in it may drop it."""
    pcb = _bend_on_pad_b(13.55)
    collapse_strict_redundant([], pcb, {1})
    assert _has_leg(pcb, 13.55), "the collapse left pad B on a shallow bend"
    assert not _shallow(pcb), _shallow(pcb)
    pcb = _bend_on_pad_b(13.8)
    collapse_strict_redundant([], pcb, {1})
    assert not _shallow(pcb), _shallow(pcb)
    print("PASS strict collapse keeps the leg from a shallow bend")


def _route_cfg():
    from routing_config import GridRouteConfig
    return GridRouteConfig(track_width=W, clearance=0.2, grid_step=0.1,
                           layers=['F.Cu', 'B.Cu'])


def test_seeded_cell_welds_to_the_pad_centre():
    """#479: a route ending on a seeded on-pad cell (the centre's cell was
    blocked by the map's conservative stamps) welds to the pad CENTRE when the
    last leg clears exactly -- not edge-on to the cell, a shallow entry."""
    from single_ended_routing import _seeded_pad_weld
    pcb = _board([])
    wx, wy = _seeded_pad_weld(pcb, 1, _route_cfg(), 10.4, 10.0, 'F.Cu')
    assert (wx, wy) == (10.0, 10.0), (wx, wy)
    print("PASS seeded cell -> welds to the pad centre")


def test_seeded_cell_keeps_its_weld_when_the_leg_is_refused():
    """Foreign copper between the cell and the centre: the exact check refuses
    the leg, and the route welds to the cell as before (#1777)."""
    from single_ended_routing import _seeded_pad_weld
    pcb = _board([((10.2, 9.0), (10.2, 11.0), 0.1, 3)])    # foreign, across
    wx, wy = _seeded_pad_weld(pcb, 1, _route_cfg(), 10.45, 10.0, 'F.Cu')
    assert (wx, wy) == (10.45, 10.0), (wx, wy)
    print("PASS refused leg -> the cell weld stands")


def test_route_passing_a_pad_welds_to_its_centre():
    """A route from pad A to pad C passes pad B 0.05 mm in, off its axis (the
    grid line beside an off-grid pad): the conversion lays a leg from the
    track to B's centre, so B is entered, not grazed."""
    from single_ended_routing import _weld_passed_pads
    pcb = _board3([])
    route = _board3([((10.0, 10.0), (10.45, 11.0), W, 1),
                     ((10.45, 11.0), (10.45, 17.0), W, 1),
                     ((10.45, 17.0), (10.0, 18.0), W, 1)]).segments
    segs = _weld_passed_pads(route, [], pcb, 1, _route_cfg())
    pcb.segments.extend(segs)
    assert not _shallow(pcb), _shallow(pcb)
    assert any(abs(s.end_x - 10.0) < 1e-6 and abs(s.end_y - 14.0) < 1e-6
               for s in segs), [(s.start_x, s.start_y, s.end_x, s.end_y) for s in segs]
    print("PASS route passing a pad -> leg to its centre")


def test_passed_pad_weld_is_refused_by_foreign_copper():
    """Foreign copper between the track and pad B's centre: the leg would not
    clear, so the route is left as it was."""
    from single_ended_routing import _weld_passed_pads
    pcb = _board3([((10.2, 13.0), (10.2, 15.0), 0.1, 3)])   # foreign, along B
    route = _board3([((10.0, 10.0), (10.45, 11.0), W, 1),
                     ((10.45, 11.0), (10.45, 17.0), W, 1),
                     ((10.45, 17.0), (10.0, 18.0), W, 1)]).segments
    segs = _weld_passed_pads(route, [], pcb, 1, _route_cfg())
    assert len(segs) == 3, [(s.start_x, s.start_y, s.end_x, s.end_y) for s in segs]
    print("PASS passed-pad leg refused -> route unchanged")


TESTS = [test_depth_geometry, test_end_on_another_track_is_not_a_pad_entry,
         test_edge_end_is_shallow, test_corner_graze_is_shallow,
         test_deep_landing_is_compliant, test_narrow_pad_needs_the_medial_axis,
         test_collapse_does_not_trim_short_of_the_pad,
         test_collapse_still_trims_a_deep_leg,
         test_collapse_drops_the_stub_not_the_entry,
         test_sweep_does_not_trim_short_of_the_pad,
         test_graze_prune_does_not_trim_short_of_the_pad,
         test_seeded_cell_welds_to_the_pad_centre,
         test_seeded_cell_keeps_its_weld_when_the_leg_is_refused,
         test_restored_stub_into_the_pad_is_adopted,
         test_microshift_does_not_slide_an_end_off_the_pad_axis,
         test_passing_track_is_shallow,
         test_collapse_does_not_leave_a_shallow_bend,
         test_route_passing_a_pad_welds_to_its_centre,
         test_passed_pad_weld_is_refused_by_foreign_copper]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(TESTS)} passed")
