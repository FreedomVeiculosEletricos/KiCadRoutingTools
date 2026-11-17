"""close_soft_joints' via->pad graze bridge (#470 class 1) skips a joint a
staircase dogbone already serves.

Before: the joint counted as served only when ONE segment ran from the via
barrel into the pad. A tap or rescue dogbone is an A* path, emitted as a
staircase of grid segments, so the pass laid a straight via->pad stub on top
of it: rescue escapes and plane taps shipped duplicate copper.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'py_router'))

from kicad_parser import Segment  # noqa: E402
from pcb_modification import _via_pad_served  # noqa: E402

PAD = SimpleNamespace(global_x=0.0, global_y=0.0)
VIA = (0.0, 1.2, 0.35)                     # x, y, barrel radius


def _pad_dist(x, y, pad):
    """A round pad of radius 0.4: 0 inside, distance to the edge outside."""
    return max(0.0, math.hypot(x - pad.global_x, y - pad.global_y) - 0.4)


def _seg(x1, y1, x2, y2, layer='F.Cu'):
    return Segment(start_x=x1, start_y=y1, end_x=x2, end_y=y2, width=0.3,
                   layer=layer, net_id=1)


def _staircase(step=0.05):
    pts = [(0.0, 1.2 - i * step) for i in range(int(1.2 / step) + 1)]
    return [_seg(*a, *b) for a, b in zip(pts, pts[1:])]


def test_single_stub_is_served():
    assert _via_pad_served([_seg(0.0, 1.2, 0.0, 0.0)], 'F.Cu', *VIA, PAD, _pad_dist)


def test_staircase_dogbone_is_served():
    assert _via_pad_served(_staircase(), 'F.Cu', *VIA, PAD, _pad_dist)


def test_no_copper_or_other_layer_is_not_served():
    assert not _via_pad_served([], 'F.Cu', *VIA, PAD, _pad_dist)
    assert not _via_pad_served(_staircase(), 'B.Cu', *VIA, PAD, _pad_dist)


def test_long_detour_is_not_a_dogbone():
    # Barrel -> 5 mm away -> back into the pad: other copper, not a local stub.
    segs = [_seg(0.0, 1.2, 5.0, 1.2), _seg(5.0, 1.2, 5.0, 0.0), _seg(5.0, 0.0, 0.0, 0.0)]
    assert not _via_pad_served(segs, 'F.Cu', *VIA, PAD, _pad_dist)


TESTS = [test_single_stub_is_served, test_staircase_dogbone_is_served,
         test_no_copper_or_other_layer_is_not_served, test_long_detour_is_not_a_dogbone]

if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f"{len(TESTS)} passed")
