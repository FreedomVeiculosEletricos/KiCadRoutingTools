#!/usr/bin/env python3
"""Grid snap and row/column alignment of placed parts (grid_align.py): each
part goes where most of its pads land on the grid, and neighbours a fraction
of a mm out of line are lined up -- only where the part stays legal.

Plain python3-runnable, no pytest."""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'py_placer'))
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

from placement.grid_align import (pad_phase, snap_candidates, on_grid,  # noqa: E402
                                  alignment_groups, shared_coordinate,
                                  snap_and_align)


class Part:
    """A two-pad part: pads at +-half along x (rot 0) or y (rot 90)."""

    def __init__(self, x, y, rot=0.0, half=0.75, w=1.6, h=0.8, side='F'):
        self.x, self.y, self.rot, self.half = x, y, rot, half
        self.w, self.h, self.side = w, h, side

    def _pad_offsets(self, rot):
        r = math.radians(-rot)
        return [(round(d * math.cos(r), 9), round(-d * math.sin(r), 9), 1)
                for d in (-self.half, self.half)]

    def rect(self):
        w, h = (self.w, self.h) if self.rot % 180 == 0 else (self.h, self.w)
        return (self.x - w / 2, self.y - h / 2, self.x + w / 2, self.y + h / 2)


def _pads_on_grid(P, grid):
    return all(abs(v / grid - round(v / grid)) < 1e-6
               for dx, dy, _ in P._pad_offsets(P.rot) for v in (P.x + dx, P.y + dy))


def test_pad_phase():
    # pads at +-0.75 on a 0.1 grid: the origin sits half a step off it
    assert pad_phase([(-0.75, 0, 1), (0.75, 0, 1)], 0.1) == (0.05, 0.0)
    assert pad_phase([(-0.8, 0, 1), (0.8, 0, 1)], 0.1) == (0.0, 0.0)
    # the majority of pads wins
    assert pad_phase([(0.0, 0, 1), (0.1, 0, 1), (0.33, 0, 1)], 0.1)[0] == 0.0
    assert pad_phase([], 0.1) == (0.0, 0.0)
    print("PASS pad phase: the offset that puts the most pads on the grid")


def test_snap_candidates_nearest_first():
    c = snap_candidates(10.03, 5.07, (0.0, 0.0), 0.1, rings=1)
    assert c[0] == (10.0, 5.1), c
    assert len(c) == 4 and set(c) == {(10.0, 5.0), (10.0, 5.1), (10.1, 5.0), (10.1, 5.1)}
    c2 = snap_candidates(10.03, 5.07, (0.0, 0.0), 0.1)
    assert len(c2) == 16 and set(c) == set(c2[:4]), c2
    assert on_grid(10.05, 5.0, (0.05, 0.0), 0.1)
    assert not on_grid(10.03, 5.0, (0.0, 0.0), 0.1)
    print("PASS snap candidates: the cell's corners first, then the next ring")


def test_snap_puts_pads_on_grid():
    parts = {'R1': Part(10.0317, 20.0442), 'R2': Part(30.0123, 7.777, rot=90.0)}
    rep = snap_and_align(parts, ['R1', 'R2'], [], grid=0.1, align_mm=0,
                         try_pose=lambda r, x, y: _set(parts, r, x, y), log=lambda s: None)
    assert rep['snapped'] == 2, rep
    for P in parts.values():
        assert _pads_on_grid(P, 0.1), (P.x, P.y)
    print("PASS snap: every pad on the grid, " + ", ".join(
        f"{r} ({P.x:g}, {P.y:g})" for r, P in parts.items()))


def test_illegal_snap_is_left_and_named():
    parts = {'R1': Part(10.0317, 20.0442)}
    rep = snap_and_align(parts, ['R1'], [], grid=0.1, align_mm=0,
                         try_pose=lambda r, x, y: False, log=lambda s: None)
    assert rep['snapped'] == 0 and rep['off_grid'] == ['R1'], rep
    assert (parts['R1'].x, parts['R1'].y) == (10.0317, 20.0442)
    print("PASS a part with no legal grid pose stays and is named")


def test_column_is_lined_up_on_the_fixed_part():
    # a column of three vertical resistors, the middle one fixed
    parts = {'R1': Part(88.2, 120.0, rot=90), 'R2': Part(88.1, 122.5, rot=90),
             'R3': Part(87.9, 125.0, rot=90), 'C1': Part(95.0, 122.5, rot=90)}
    rep = snap_and_align(parts, ['R1', 'R3', 'C1'], ['R2'], grid=0.1, align_mm=0.5,
                         try_pose=lambda r, x, y: _set(parts, r, x, y), log=lambda s: None)
    assert [P.x for r, P in sorted(parts.items()) if r != 'C1'] == [88.1, 88.1, 88.1], parts
    assert parts['C1'].x == 95.0
    assert rep['aligned'] == [{'axis': 'x', 'at': 88.1, 'refs': ['R1', 'R3']}], rep
    print(f"PASS column lined up on the fixed part: {rep['aligned']}")


def test_groups_need_same_face_rotation_and_reach():
    items = {'A': (10.0, 10.0, (9.6, 9.2, 10.4, 10.8), 'F', 90.0),
             'B': (10.2, 12.0, (9.8, 11.2, 10.6, 12.8), 'F', 270.0),   # 180 apart: pairs
             'C': (10.1, 14.0, (9.7, 13.2, 10.5, 14.8), 'B', 90.0),    # other face
             'D': (10.3, 30.0, (9.9, 29.2, 10.7, 30.8), 'F', 90.0),    # too far along
             'E': (10.6, 10.0, (10.2, 9.2, 11.0, 10.8), 'F', 0.0)}     # other rotation
    g = alignment_groups(items, tol=0.5, reach=2.0)
    assert g == [('x', ['A', 'B'])], g
    assert shared_coordinate([1.0, 1.2, 1.25]) == 1.2
    assert shared_coordinate([1.0, 1.2], fixed=[1.0]) == 1.0
    print("PASS groups: one face, one rotation mod 180, within reach")


def test_alignment_that_breaks_a_limit_is_undone():
    parts = {'R1': Part(10.0, 10.0, rot=90), 'R2': Part(10.3, 12.0, rot=90)}

    def limits():
        return (1 if parts['R2'].x == 10.0 or parts['R1'].x == 10.3 else 0,)
    rep = snap_and_align(parts, ['R1', 'R2'], [], grid=0.1, align_mm=0.5,
                         try_pose=lambda r, x, y: _set(parts, r, x, y), limits=limits,
                         log=lambda s: None)
    assert rep['aligned'] == [] and (parts['R1'].x, parts['R2'].x) == (10.0, 10.3), rep
    print("PASS a line-up that breaks a decap/claim limit is undone")


def test_second_pass_snaps_a_part_its_neighbour_freed():
    """A is pressed against B (gap rule 1.0 mm between centres); A comes
    first and has no room until B has snapped away."""
    parts = {'A': Part(10.0, 0.04, half=0.0), 'B': Part(11.0, 0.03, half=0.0)}
    parts['B'].x = 10.96

    def try_pose(r, x, y):
        o = parts['B' if r == 'A' else 'A']
        if abs(x - o.x) < 1.0 - 1e-9 or (r == 'B' and x < 11.0 - 1e-9):
            return False
        if r == 'A' and abs(x - 10.0) > 1e-9:   # A has room only at x = 10
            return False
        return _set(parts, r, x, y)
    # A at 10.04 x -> its grid poses within 2 steps all sit < 1.0 from B (11.0)
    parts['A'].x = 10.04
    rep = snap_and_align(parts, ['A', 'B'], [], grid=0.1, align_mm=0,
                         try_pose=try_pose, log=lambda s: None)
    assert rep['off_grid'] == [] and rep['snapped'] == 2, rep
    print(f"PASS a second pass snaps A after B made room: A ({parts['A'].x:g}), "
          f"B ({parts['B'].x:g})")


def _set(parts, r, x, y):
    parts[r].x, parts[r].y = x, y
    return True


TESTS = [test_pad_phase, test_snap_candidates_nearest_first, test_snap_puts_pads_on_grid,
         test_illegal_snap_is_left_and_named, test_column_is_lined_up_on_the_fixed_part,
         test_groups_need_same_face_rotation_and_reach,
         test_alignment_that_breaks_a_limit_is_undone,
         test_second_pass_snaps_a_part_its_neighbour_freed]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(TESTS)} passed")
