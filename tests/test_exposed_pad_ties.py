#!/usr/bin/env python3
"""A package's same-net pins are tied straight to its exposed pad before
routing (exposed_pad_ties.py), and route.py's end-of-run pass never ships a
merge of its two candidates that puts one net's copper on another's
(_late_merge_clears).

Plain python3-runnable, no pytest."""
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from kicad_parser import parse_kicad_pcb, Segment  # noqa: E402
from exposed_pad_ties import (exposed_pads, tie_pins_to_exposed_pads,  # noqa: E402
                              plan_exposed_pad_ties)

# A 12-pin QFN at (10, 10): a 3 x 3 exposed pad (net 1) and three pins a
# side at 0.5 mm pitch, 0.25 x 0.8, centred 2.1 mm out. Pins 2 (bottom
# middle) and 5 (right middle) are net 1; pin 1 (bottom left, outside the
# exposed pad's span) is net 1 too; the rest are net 2.
_PINS = [  # number, x, y, size_x, size_y, net
    (1, 9.5, 12.1, 0.25, 0.8, 1), (2, 10.0, 12.1, 0.25, 0.8, 1),
    (3, 10.5, 12.1, 0.25, 0.8, 2),
    (4, 12.1, 10.5, 0.8, 0.25, 2), (5, 12.1, 10.0, 0.8, 0.25, 1),
    (6, 12.1, 9.5, 0.8, 0.25, 2),
    (7, 10.5, 7.9, 0.25, 0.8, 2), (8, 10.0, 7.9, 0.25, 0.8, 2),
    (9, 9.5, 7.9, 0.25, 0.8, 2),
    (10, 7.9, 9.5, 0.8, 0.25, 2), (11, 7.9, 10.0, 0.8, 0.25, 2),
    (12, 7.9, 10.5, 0.8, 0.25, 2),
]


def _board(segs=()):
    pads = "".join(
        f'\t\t(pad "{n}" smd rect (at {x - 10:g} {y - 10:g}) (size {sx} {sy})\n'
        f'\t\t\t(layers "F.Cu" "F.Mask") (net {net} "N{net}")\n'
        f'\t\t\t(uuid "aaaaaaaa-0000-0000-0000-0000000000{n:02d}"))\n'
        for n, x, y, sx, sy, net in _PINS)
    body = "".join(
        f'\t(segment (start {a} {b}) (end {c} {d}) (width 0.15) (layer "F.Cu")\n'
        f'\t\t(net {net}) (uuid "bbbbbbbb-0000-0000-0000-0000000{i:05d}"))\n'
        for i, (a, b, c, d, net) in enumerate(segs))
    text = f"""(kicad_pcb
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
	(footprint "test:QFN12"
		(layer "F.Cu")
		(uuid "aaaaaaaa-0000-0000-0000-000000000001")
		(at 10 10)
		(pad "13" smd rect (at 0 0) (size 3 3)
			(layers "F.Cu" "F.Mask") (net 1 "N1")
			(uuid "aaaaaaaa-0000-0000-0000-0000000000ff"))
{pads}	)
{body})
"""
    d = tempfile.mkdtemp(prefix="eptie_")
    p = os.path.join(d, "b.kicad_pcb")
    with open(p, "w") as f:
        f.write(text)
    return parse_kicad_pcb(p)


def _cfg():
    return SimpleNamespace(clearance=0.15, track_width=0.15, net_clearances={},
                           board_edge_clearance=0.0, hole_to_hole_clearance=0.2,
                           grid_step=0.1)


def _ends(ties):
    return sorted((round(t.start_x, 3), round(t.start_y, 3),
                   round(t.end_x, 3), round(t.end_y, 3)) for t in ties)


def test_exposed_pad_is_found():
    pcb = _board()
    found = exposed_pads(pcb)
    assert len(found) == 1, found
    eps = next(iter(found.values()))[0]
    assert [(e.size_x, e.size_y) for e in eps] == [(3.0, 3.0)]
    print("PASS the exposed pad is found")


def test_facing_pins_are_tied_straight_inward():
    pcb = _board()
    ties = tie_pins_to_exposed_pads(pcb, {1}, _cfg())
    # pin 2 goes up into the pad, pin 5 left into it; each ends one track
    # width inside the pad's edge (y 11.5 / x 11.5). Pin 1 at x 9.5 is
    # inside the pad's span too (8.5..11.5).
    assert _ends(ties) == [(9.5, 12.1, 9.5, 11.35), (10.0, 12.1, 10.0, 11.35),
                           (12.1, 10.0, 11.35, 10.0)], _ends(ties)
    assert all(t.net_id == 1 and t.width == 0.15 for t in ties)
    assert all(t in pcb.segments for t in ties)
    print("PASS facing pins are tied straight inward")


def test_other_nets_and_tied_pins_are_left_alone():
    pcb = _board(segs=[(10.0, 12.1, 10.0, 11.0, 1)])        # pin 2 already tied
    ties = tie_pins_to_exposed_pads(pcb, {1, 2}, _cfg())
    assert (10.0, 12.1, 10.0, 11.35) not in _ends(ties), _ends(ties)
    assert all(t.net_id == 1 for t in ties), "net 2 has no exposed pad"
    assert not tie_pins_to_exposed_pads(_board(), {2}, _cfg())
    print("PASS other nets and already tied pins are left alone")


def test_a_stub_that_would_not_clear_is_not_laid():
    # A net-3 track across the strip between the bottom row and the pad.
    pcb = _board(segs=[(9.0, 11.8, 11.0, 11.8, 3)])
    ties = tie_pins_to_exposed_pads(pcb, {1}, _cfg())
    assert _ends(ties) == [(12.1, 10.0, 11.35, 10.0)], _ends(ties)
    print("PASS a stub that would not clear is not laid")


def test_the_planner_asks_the_gate_once_per_tie():
    pcb = _board()
    calls = []

    def clears(*a):
        calls.append(a)
        return True
    ties = plan_exposed_pad_ties(pcb, {1}, 0.15, 0.15, clears)
    assert len(ties) == 3 and len(calls) == 3
    print("PASS the planner asks the gate once per tie")


def test_merge_of_candidates_reverts_a_net_that_would_short():
    from route import _late_merge_clears
    pcb = _board()
    # Before: net 1 and net 3 each have a stub far apart.
    n1 = Segment(start_x=20, start_y=20, end_x=21, end_y=20, width=0.15,
                 layer='F.Cu', net_id=1)
    n3 = Segment(start_x=20, start_y=22, end_x=21, end_y=22, width=0.15,
                 layer='F.Cu', net_id=3)
    before = [n1, n3]
    # A: net 3 rebuilt onto y=21. B: net 1 rebuilt onto y=21 too (clear of
    # B's own net-3 copper, which stayed at y=22).
    n3a = Segment(start_x=20, start_y=21, end_x=21, end_y=21, width=0.15,
                  layer='F.Cu', net_id=3)
    n1b = Segment(start_x=20, start_y=21.05, end_x=21, end_y=21.05, width=0.15,
                  layer='F.Cu', net_id=1)
    segs_a = list(pcb.segments) + [n1, n3a]
    segs_b = list(pcb.segments) + [n1b, n3]
    keep, n = _late_merge_clears(pcb, segs_a, segs_b, {1}, before, _cfg())
    assert keep == set() and n == 1, (keep, n)
    # B kept where it clears A's copper.
    n1c = Segment(start_x=20, start_y=19, end_x=21, end_y=19, width=0.15,
                  layer='F.Cu', net_id=1)
    keep, n = _late_merge_clears(pcb, segs_a, list(pcb.segments) + [n1c, n3],
                                 {1}, before, _cfg())
    assert keep == {1} and n == 0, (keep, n)
    print("PASS the candidate merge reverts a net that would short")


TESTS = [test_exposed_pad_is_found, test_facing_pins_are_tied_straight_inward,
         test_other_nets_and_tied_pins_are_left_alone,
         test_a_stub_that_would_not_clear_is_not_laid,
         test_the_planner_asks_the_gate_once_per_tie,
         test_merge_of_candidates_reverts_a_net_that_would_short]

if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f"{len(TESTS)} passed")
