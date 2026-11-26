"""Plane region joins obey --escalation per net.

Before: repair_planes (and route.py's in-run plane finalize) joined plane
regions from the 0.2 mm REPAIR_MIN_TRACK_WIDTH up, and its last-resort bridge
(#217) at the run's --track-width, whatever the policy. Under
`--escalation off` a GND plane asked at 0.5 mm shipped 0.2-0.3 mm straps
(FRE111R14-A). Now each plane net gets its own range: under `off` nothing
narrower than that net's width (two nets asking x and y each get theirs),
under `board` nothing below that net's rule floor, under `fab` unchanged.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'py_router'))

import fab_tiers  # noqa: E402
from kicad_parser import parse_kicad_pcb  # noqa: E402
from repair_planes import plane_join_widths, _plane_join_width_resolver  # noqa: E402
from routing_config import GridRouteConfig  # noqa: E402


def test_off_keeps_each_nets_own_width():
    # GND asks 0.5, VS1 asks 1.0, a fine net asks 0.15 (below the 0.2 default).
    assert plane_join_widths('off', 0.2, 2.0, 0.3, 0.5) == (0.5, 2.0, 0.5)
    assert plane_join_widths('off', 0.2, 2.0, 0.3, 1.0) == (1.0, 2.0, 1.0)
    # Widening is not narrowing: start at 0.2, last resort at the net's 0.15.
    assert plane_join_widths('off', 0.2, 2.0, 0.3, 0.15) == (0.2, 2.0, 0.15)
    # A net wider than --max-track-width still gets its own width.
    assert plane_join_widths('off', 0.2, 2.0, 0.3, 2.5) == (2.5, 2.5, 2.5)


def test_board_stops_at_the_rule_floor():
    assert plane_join_widths('board', 0.2, 2.0, 0.3, 0.5, 0.25) == (0.25, 2.0, 0.3)
    assert plane_join_widths('board', 0.2, 2.0, 0.3, 0.5, 0.4) == (0.4, 2.0, 0.4)
    assert plane_join_widths('board', 0.2, 2.0, 0.3, 0.5, 0.0) == (0.2, 2.0, 0.3)


def test_fab_is_unchanged():
    assert plane_join_widths('fab', 0.2, 2.0, 0.3, 0.5, 0.4) == (0.2, 2.0, 0.3)


def test_resolver_reads_power_and_class_widths_per_net():
    cfg = GridRouteConfig(track_width=0.3, layers=['F.Cu', 'In1.Cu', 'B.Cu'])
    cfg.power_net_widths = {1: 0.5}
    prev = fab_tiers.get_escalation_policy()
    try:
        fab_tiers.set_escalation_policy('off')
        resolve = _plane_join_width_resolver(
            cfg, None, '', [1, 2, 3], {2: 0.4}, {}, 0.2, 2.0)
        assert resolve(1, 'In1.Cu') == (0.5, 2.0, 0.5)    # power width
        assert resolve(2, 'In1.Cu') == (0.4, 2.0, 0.4)    # class width
        assert resolve(3, 'In1.Cu') == (0.3, 2.0, 0.3)    # --track-width
        # the caller's config is left alone
        assert cfg.net_track_widths == {}
        fab_tiers.set_escalation_policy('fab')
        resolve = _plane_join_width_resolver(
            cfg, None, '', [1, 2, 3], {2: 0.4}, {}, 0.2, 2.0)
        assert resolve(1, 'In1.Cu') == (0.2, 2.0, 0.3)
    finally:
        fab_tiers.set_escalation_policy(*prev)


BOARD = (Path(__file__).resolve().parent.parent / 'kicad_files'
         / 'lvds_converter_dualclk_gnd.kicad_pcb')
DRU = """(version 1)
(rule "gnd"
  (constraint track_width (min 0.3mm) (opt 0.5mm))
  (condition "A.NetName == '/GND'"))
"""


def test_resolver_reads_the_board_rules():
    # net_track_widths None (the standalone CLI, --track-width omitted): under
    # off the joins read the .kicad_dru opt; under board, its min is the floor.
    td = tempfile.mkdtemp()
    prev = fab_tiers.get_escalation_policy()
    try:
        board = os.path.join(td, 'b.kicad_pcb')
        shutil.copy(BOARD, board)
        with open(os.path.join(td, 'b.kicad_dru'), 'w') as f:
            f.write(DRU)
        pcb = parse_kicad_pcb(board)
        gnd = next(n for n, net in pcb.nets.items() if net.name == '/GND')
        other = next(n for n in pcb.nets if n not in (0, gnd))
        layers = list(pcb.board_info.copper_layers)
        cfg = GridRouteConfig(track_width=0.25, layers=layers)
        fab_tiers.set_escalation_policy('off')
        resolve = _plane_join_width_resolver(
            cfg, pcb, board, [gnd, other], None, None, 0.2, 2.0)
        assert resolve(gnd, 'B.Cu') == (0.5, 2.0, 0.5)
        fab_tiers.set_escalation_policy('board', {'track_width': 0.15})
        resolve = _plane_join_width_resolver(
            cfg, pcb, board, [gnd, other], None, None, 0.2, 2.0)
        assert resolve(gnd, 'B.Cu') == (0.3, 2.0, 0.3)
        assert resolve(other, 'B.Cu') == (0.2, 2.0, 0.25)
    finally:
        fab_tiers.set_escalation_policy(*prev)
        shutil.rmtree(td, ignore_errors=True)


TESTS = [test_off_keeps_each_nets_own_width, test_board_stops_at_the_rule_floor,
         test_fab_is_unchanged, test_resolver_reads_power_and_class_widths_per_net,
         test_resolver_reads_the_board_rules]

if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f"{len(TESTS)} passed")
