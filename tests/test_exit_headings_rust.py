#!/usr/bin/env python3
"""Rust A*: SMD exit mouths as terminals (route_multi source_headings /
target_headings, grid_router 0.23.8).

A source with a heading is the mouth of a straight exit the track already
runs along: its first move turns at most 90 degrees. A target with a heading
is entered by a move turning at most 90 degrees onto the heading the track
continues along into the pad. extra_g is the exit's own length, so between
two mouths the search takes the cheaper WHOLE route. No headings: the search
is unchanged.

    python3 tests/test_exit_headings_rust.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rust_router'))

from grid_router import GridObstacleMap, GridRouter  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def route(src, tgt, sh=None, th=None, obs=None):
    obs = obs or GridObstacleMap(1)
    r = GridRouter(via_cost=5000, h_weight=1.0, turn_cost=1000)
    path, _it, _st = r.route_multi(obs, src, tgt, 20000,
                                   source_headings=sh, target_headings=th)
    return path


def step(path, i):
    (ax, ay, _), (bx, by, _) = path[i], path[i + 1]
    return (bx - ax, by - ay)


def main():
    print("1. a source mouth's first move turns at most 90 degrees")
    free = route([(0, 0, 0)], [(-10, 0, 0)])
    check("control: with no heading the route runs straight back",
          free and step(free, 0) == (-1, 0), free and step(free, 0))
    p = route([(0, 0, 0)], [(-10, 0, 0)], sh=[(0, 0, 0, 1, 0, 0)])
    first = step(p, 0) if p else None
    check("heading +x: the first move is not backwards",
          p is not None and first[0] >= 0, first)

    print("2. a target mouth is entered at most 90 degrees onto its heading")
    free = route([(10, 6, 0)], [(10, 0, 0)])
    check("control: with no heading the route arrives moving down",
          free and step(free, len(free) - 2)[1] < 0, free and step(free, len(free) - 2))
    p = route([(10, 6, 0)], [(10, 0, 0)], th=[(10, 0, 0, 0, 1, 0)])
    last = step(p, len(p) - 2) if p else None
    check("heading +y: the last move does not run against it",
          p is not None and last[1] >= 0, last)

    print("3. extra_g: the cheaper whole route picks the mouth")
    tg = [(10, 3, 0), (10, -3, 0)]
    p = route([(0, 0, 0)], tg, th=[(10, 3, 0, 1, 0, 50000), (10, -3, 0, 1, 0, 0)])
    check("the mouth whose exit is short wins", p is not None and p[-1][:2] == (10, -3),
          p and p[-1])
    p = route([(0, 0, 0)], tg, th=[(10, 3, 0, 1, 0, 0), (10, -3, 0, 1, 0, 50000)])
    check("...and the other way round", p is not None and p[-1][:2] == (10, 3),
          p and p[-1])
    p = route([(0, 0, 0), (0, 8, 0)], [(20, 8, 0)],
              sh=[(0, 0, 0, 1, 0, 0), (0, 8, 0, 1, 0, 90000)])
    check("a source mouth's extra_g counts too", p is not None and p[0][:2] == (0, 0),
          p and p[0])

    print("4. no headings: unchanged")
    a = route([(0, 0, 0)], [(7, 4, 0)])
    b = route([(0, 0, 0)], [(7, 4, 0)], sh=[], th=[])
    check("empty headings give the same path", a == b and a is not None)

    if FAILS:
        print(f"\nFAILED {len(FAILS)}: " + "; ".join(FAILS))
        sys.exit(1)
    print("\nALL PASS")


if __name__ == '__main__':
    main()
