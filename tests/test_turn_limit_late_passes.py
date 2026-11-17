#!/usr/bin/env python3
"""The passes that lay copper after the router keep its corner limit
(--max-turn-angle / --min-corner-angle).

Plain python3-runnable, no pytest."""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from pcb_modification import _min_corner, _bend_too_sharp, _creates_acute  # noqa: E402
from kicad_parser import Segment  # noqa: E402


def _cfg(max_turn=180.0, min_corner=90.0):
    from routing_config import GridRouteConfig
    return GridRouteConfig(max_turn_angle=max_turn, min_corner_angle=min_corner)


def test_min_corner_follows_the_turn_limit():
    assert _min_corner(_cfg()) == 90.0
    assert _min_corner(_cfg(max_turn=45)) == 135.0
    assert _min_corner(_cfg(max_turn=90)) == 90.0
    assert _min_corner(_cfg(min_corner=0)) == 90.0     # off: the old right angle
    assert _min_corner(None) == 90.0
    print("PASS the passes' corner limit follows --max-turn-angle")


def test_bend_test_matches_the_old_one_at_90():
    # heading east, then north (90), north-east (45), west (180), south-west (135)
    for v, sharp90 in (((0, 1), False), ((1, 1), False), ((-1, 0), True),
                       ((-1, -1), True), ((-1, 1), True)):
        assert _bend_too_sharp(1, 0, *v, 90.0) == sharp90, v
        assert _bend_too_sharp(1, 0, *v, 90.0) == (1 * v[0] + 0 * v[1] < -1e-9)
    print("PASS at 90 degrees the bend test is the old dot-product test")


def test_bend_test_at_135_allows_only_45_degree_turns():
    assert not _bend_too_sharp(1, 0, 1, 0, 135.0)       # straight on
    assert not _bend_too_sharp(1, 0, 1, 1, 135.0)       # 45 turn
    assert _bend_too_sharp(1, 0, 0, 1, 135.0)           # 90 turn
    assert _bend_too_sharp(1, 0, -1, 1, 135.0)          # 135 turn
    print("PASS at 135 degrees interior only 45-degree turns pass")


def test_creates_acute_takes_the_limit():
    # an existing track runs north from (0, 0); a new run leaves (0, 0) east
    north = Segment(start_x=0, start_y=0, end_x=0, end_y=-1, width=0.15,
                    layer='F.Cu', net_id=1)
    east = [(0, 0), (1, 0)]
    assert not _creates_acute(east, 'F.Cu', [north], min_angle=90.0)
    assert _creates_acute(east, 'F.Cu', [north], min_angle=135.0)
    print("PASS a right-angle joint is refused under a 135-degree limit")


def test_final_corner_pass_is_off_at_the_default_limit():
    from route import _late_final_corners
    # at the default limit the pass returns before touching the board
    assert _late_final_corners(None, None, False, None, None, None, {'X'},
                               False, _cfg()) == 0
    print("PASS the last corner pass does not run at the default limit")


def test_an_acute_corner_gets_two_cuts_under_a_45_degree_limit():
    from corner_angle import (corner_cut_candidates, corner_cut_chains,
                              interior_angle)
    # the shipped /SWDIO corner: in along the diagonal, out straight up
    V, A, C = (64.1, 43.9), (64.4, 43.6), (64.1, 43.1)
    # one cut needs a bend at A or C itself (graded by the caller against
    # the tracks beyond); every cut it offers inside the legs fails
    assert all(P == A or Q == C for P, Q in
               corner_cut_candidates(V, A, C, 135.0, 0.05, width=0.15))
    chains = corner_cut_chains(V, A, C, 135.0, 0.05, width=0.15)
    assert chains, "no two-cut recorner"
    P, R, Q = chains[0]
    pts = [A, P, R, Q, C]
    for k in range(1, 4):
        a, b, c = pts[k - 1], pts[k], pts[k + 1]
        ang = interior_angle(a[0] - b[0], a[1] - b[1], c[0] - b[0], c[1] - b[1])
        assert ang >= 134.5, (k, ang, pts)
    # a right angle still takes one cut
    assert corner_cut_candidates((0, 0), (-1, 0), (0, 1), 135.0, 0.05, width=0.15)
    print("PASS an acute corner gets two cuts under a 45-degree limit")


TESTS = [test_min_corner_follows_the_turn_limit,
         test_an_acute_corner_gets_two_cuts_under_a_45_degree_limit,
         test_bend_test_matches_the_old_one_at_90,
         test_bend_test_at_135_allows_only_45_degree_turns,
         test_creates_acute_takes_the_limit,
         test_final_corner_pass_is_off_at_the_default_limit]

if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f"{len(TESTS)} passed")
