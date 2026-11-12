#!/usr/bin/env python3
"""The #1063 cleanup never cuts a track back to an annulus KiCad does not flash.

A through-hole pad with `(remove_unused_layers yes)` has no copper on the
layers its mode removes unless something reaches its HOLE; KiCad's
connectivity tests it there by the hole, not the outline. #1063's removal
model graded pad joints by the outline, so on ecp5_mini's edge headers
(0.7 mm drill, 1.0 mm pad, `keep_end_layers yes`) it cut the In1.Cu tail that
ran to the pad centre, leaving the track end 0.46 mm out: inside the outline,
outside the drill. KiCad then graded seven diff-pair nets open that
check_connected's copper grading passed (corpus A/B 2026-09-28, ecp5_mini
open nets 1 -> 8).

Pinned here:
  1. both file spellings of the mode parse (KiCad 8+ yes/no, KiCad 7 bare);
  2. which layers each mode leaves unflashed;
  3. default grading is unchanged, and the opt-in grades the annulus-only end
     open;
  4. the cleanup and check_weird keep the tail on an unflashed layer, and
     still remove it where the pad is flashed (the #1063 contract).

    python3 tests/test_1063_unflashed_pad_hole_joint.py
"""
import collections
import os
import sys
from types import SimpleNamespace

_TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_TESTS)
for p in (ROOT, _TESTS, os.path.join(ROOT, 'py_router'), os.path.join(ROOT, 'py_tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

from synth import make_pad, make_seg, make_pcb, make_net
from kicad_parser import unconnected_layer_mode_from_text
from connectivity import pad_unflashed_layers, endpoint_reaches_pad
from check_connected import check_net_connectivity
from check_weird import check_weird
from pcb_modification import collapse_strict_redundant

FAILS = []
NET = 7
LAYERS = ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu']
TAIL_END = (0.32, 0.33)   # 0.46 mm from the pad centre: past the 0.35 drill


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  {detail}" if detail else ""))


def _pads(mode):
    """ecp5_mini J3's header pad: 1.0 mm round, 0.7 mm drill."""
    return [make_pad(NET, 0, 0, ref='J3', num='12', size_x=1.0, size_y=1.0,
                     shape='circle', layers=('*.Cu', '*.Mask'), drill=0.7,
                     pad_type='thru_hole', unconnected_layer_mode=mode),
            make_pad(NET, 6, 0, ref='J3', num='14', size_x=1.0, size_y=1.0,
                     shape='circle', layers=('*.Cu', '*.Mask'), drill=0.7,
                     pad_type='thru_hole')]


def _board(mode, layer):
    tail = make_seg(0, 0, *TAIL_END, net_id=NET, layer=layer, width=0.1)
    trunk = make_seg(*TAIL_END, 6, 0, net_id=NET, layer=layer, width=0.1)
    pads = _pads(mode)
    pcb = make_pcb(nets={NET: make_net(NET, '/PB04-')}, segments=[tail, trunk],
                   pads_by_net={NET: pads},
                   board_info=SimpleNamespace(copper_layers=list(LAYERS)))
    return pcb, tail, trunk


def parsing():
    print("1. the mode parses from both file spellings")
    cases = [
        ('', 'keep_all'),
        ('(remove_unused_layers no)', 'keep_all'),
        ('(remove_unused_layers yes) (keep_end_layers yes)', 'remove_except_start_end'),
        ('(remove_unused_layers yes) (keep_end_layers no)', 'remove_all'),
        ('(remove_unused_layers yes)', 'remove_all'),
        ('(remove_unused_layers) (keep_end_layers)', 'remove_except_start_end'),
    ]
    for text, want in cases:
        got = unconnected_layer_mode_from_text(f'(pad "1" thru_hole circle {text})')
        check(f"{text or '<none>'} -> {want}", got == want, got)


def layers():
    print("2. which layers each mode leaves unflashed")
    tht = _pads('remove_except_start_end')[0]
    check("remove_except_start_end: the inner layers only",
          pad_unflashed_layers(tht, LAYERS) == {'In1.Cu', 'In2.Cu'})
    tht.unconnected_layer_mode = 'remove_all'
    check("remove_all: every layer", pad_unflashed_layers(tht, LAYERS) == set(LAYERS))
    tht.unconnected_layer_mode = 'keep_all'
    check("keep_all: none", pad_unflashed_layers(tht, LAYERS) == set())
    smd = make_pad(NET, 0, 0, unconnected_layer_mode='remove_all')
    check("an SMD pad has no hole, so no mode", pad_unflashed_layers(smd, LAYERS) == set())


def grading():
    print("3. grading: default unchanged, opt-in reads the hole")
    pcb, tail, trunk = _board('remove_except_start_end', 'In1.Cu')
    pads = pcb.pads_by_net[NET]
    r = check_net_connectivity(NET, [trunk], [], pads, [])
    check("default grading still credits the annulus end", r['connected'])
    r = check_net_connectivity(NET, [trunk], [], pads, [], unflashed_hole_only=True)
    check("the opt-in grades the annulus-only end open (KiCad's verdict)",
          not r['connected'])
    r = check_net_connectivity(NET, [tail, trunk], [], pads, [], unflashed_hole_only=True)
    check("the tail into the hole connects under the opt-in", r['connected'])
    check("endpoint_reaches_pad: outline by default",
          endpoint_reaches_pad(*TAIL_END, 0.05, ('In1.Cu',), pads[0]) == {'In1.Cu'})
    check("endpoint_reaches_pad: nothing on the unflashed layer under the opt-in",
          endpoint_reaches_pad(*TAIL_END, 0.05, ('In1.Cu',), pads[0],
                               unflashed_hole_only=True) == set())
    check("endpoint_reaches_pad: an end layer keeps the outline under the opt-in",
          endpoint_reaches_pad(*TAIL_END, 0.05, ('F.Cu',), pads[0],
                               unflashed_hole_only=True) == {'F.Cu'})


def _cleanup(mode, layer, pad_entry_rule=False):
    """The trunk ends 0.04 mm inside the pad outline, so the tail is also
    what carries it into the pad: the pad-entry rule (pad_entry.py) keeps it
    on every layer. The #1063 checks isolate the unflashed-layer joint by
    switching that rule off; ``pad_entry_rule`` grades with it on."""
    import pad_entry
    real = pad_entry.shallow_pad_entry
    if not pad_entry_rule:
        pad_entry.shallow_pad_entry = lambda *a, **k: None
    try:
        pcb, tail, trunk = _board(mode, layer)
        flagged = [f for f in check_weird(pcb)[0] if f['category'] == 'removable-segment']
        n, _ = collapse_strict_redundant([], pcb, None)
    finally:
        pad_entry.shallow_pad_entry = real
    return len(flagged), n, tail in pcb.segments, trunk in pcb.segments


def cleanup():
    print("4. the cleanup and check_weird")
    flagged, n, tail_kept, trunk_kept = _cleanup('remove_except_start_end', 'In1.Cu')
    check("unflashed In1.Cu: check_weird does not call the tail removable",
          flagged == 0, f"flagged {flagged}")
    check("unflashed In1.Cu: the cleanup keeps the tail",
          n == 0 and tail_kept and trunk_kept, f"removed {n}")
    flagged, n, tail_kept, trunk_kept = _cleanup('remove_all', 'F.Cu')
    check("remove_all on F.Cu: the tail stays too", n == 0 and tail_kept, f"removed {n}")
    # Controls: where KiCad flashes the pad, the tail is a buried wiggle and
    # #1063 removes it -- so the fixture really exercises the removal.
    flagged, n, tail_kept, trunk_kept = _cleanup('keep_all', 'In1.Cu')
    check("keep_all: the tail is removable and removed (the #1063 contract)",
          flagged == 1 and n == 1 and not tail_kept and trunk_kept,
          f"flagged {flagged}, removed {n}")
    flagged, n, tail_kept, trunk_kept = _cleanup('remove_except_start_end', 'F.Cu')
    check("an end layer keeps its pad: the F.Cu tail is removed",
          n == 1 and not tail_kept and trunk_kept, f"removed {n}")
    # With the pad-entry rule on, the tail is the trunk's way into the pad:
    # removing it would leave the trunk joined 0.04 mm deep.
    flagged, n, tail_kept, trunk_kept = _cleanup('keep_all', 'In1.Cu', True)
    check("pad-entry rule: the tail that carries the trunk into the pad stays",
          flagged == 0 and n == 0 and tail_kept, f"flagged {flagged}, removed {n}")


def main():
    parsing()
    layers()
    grading()
    cleanup()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: " + "; ".join(FAILS))
        return 1
    print("ALL PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
