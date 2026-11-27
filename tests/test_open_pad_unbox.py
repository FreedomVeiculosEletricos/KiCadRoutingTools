#!/usr/bin/env python3
"""route.py's end-of-run open-pad unboxing (open_pad_unbox.py).

A pad left off its net because another net's track passes too close to it
for a track to fit between: the pass takes that track off a copy of the
written board, reroutes both nets with the run's own engine, and ships the
copy only when it frees pads, no net loses a pad, and check_drc finds no
more of anything.

  1. boxing_branches finds the track that boxes the pad (within
     2*clearance + width), not one farther off, and never frozen copper;
  2. a reroute that frees the pad ships: the reroute was handed a board
     without the boxing track and asked for both nets;
  3. a reroute that costs the boxing net a pad is refused, file unchanged;
  4. a reroute that frees the pad but adds a DRC finding is refused;
  5. a reroute that writes nothing leaves the board as it was;
  6. locked copper and --keep-input-copper copper are never taken up;
  7. CLI: route.py prints COMPLETION:, --require-complete exits 4 with a pad
     left open, 0 when every routed net is complete, and --no-unbox skips
     the pass.

    python3 tests/test_open_pad_unbox.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for p in (ROOT, TESTS_DIR, os.path.join(ROOT, 'py_router')):
    if p not in sys.path:
        sys.path.insert(0, p)

from synth import RoutingBoard                                  # noqa: E402
from kicad_parser import parse_kicad_pcb                        # noqa: E402
from improvement_gate import copper_item_key                    # noqa: E402
from run_utils import evidence                                  # noqa: E402
import open_pad_unbox as opu                                    # noqa: E402

FAILS = []
CLEARANCE, WIDTH = 0.1, 0.15


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name
          + (f'  {detail}' if detail else ''))


def boxed_board(tmp, tag, locked=False):
    """SIG: S1 (10,15) and S2 (3,15), not joined. B: a track from B1 to B2
    at x=10.6, 0.05 mm of copper from S1's right edge."""
    rb = RoutingBoard()
    # S2 first: the grade reports ONE pad of a two-pad net open, and the
    # pass tries only the pads it reports -- S1, declared second.
    rb.pad('S2', 3, 15, 'SIG')
    rb.pad('S1', 10, 15, 'SIG')
    rb.pad('B1', 10.6, 5, 'B')
    rb.pad('B2', 10.6, 25, 'B')
    rb.seg(10.6, 5, 10.6, 25, 'B', w=0.2, locked=locked, uid='box-track')
    path = os.path.join(tmp, f'{tag}.kicad_pcb')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(rb.text())
    return rb, path


def seg_text(rb, x1, y1, x2, y2, net, w=0.2):
    return ('  (segment (start %s %s) (end %s %s) (width %s) (layer "F.Cu") '
            '(net %d) (uuid "fake-%s-%s-%s"))'
            % (x1, y1, x2, y2, w, rb.net(net), net, x1, y1))


def fake_route(rb, extra, calls, write=True):
    """A route_fn that appends `extra` segments to the board it is handed."""
    def route_fn(tin, tout, nets):
        txt = open(evidence(tin, 'the unbox input'), encoding='utf-8').read()
        calls.append((list(nets), txt))
        if not write:
            return
        body = txt.rstrip().rstrip(')')
        with open(tout, 'w', encoding='utf-8') as fh:
            fh.write(body + '\n' + '\n'.join(
                seg_text(rb, *s) for s in extra) + '\n)\n')
    return route_fn


SIG_JOIN = (10, 15, 3, 15, 'SIG')
B_DETOUR = [(10.6, 5, 11.5, 5.9, 'B'), (11.5, 5.9, 11.5, 24.1, 'B'),
            (11.5, 24.1, 10.6, 25, 'B')]
B_SHORT = [(10.6, 5, 10.45, 5.5, 'B'), (10.45, 5.5, 10.45, 24.5, 'B'),
           (10.45, 24.5, 10.6, 25, 'B')]


def run(path, rb, extra, write=True, keep_input=None):
    calls = []
    before = open(path, encoding='utf-8').read()
    freed = opu.unbox_open_pads(path, {'SIG', 'B'},
                                fake_route(rb, extra, calls, write),
                                clearance=CLEARANCE, width=WIDTH,
                                keep_input=keep_input)
    after = open(path, encoding='utf-8').read()
    return freed, calls, before, after


def open_count(path):
    b = parse_kicad_pcb(path)
    return len(opu.open_pads(b, {n for n, x in b.nets.items()
                                 if x.name in ('SIG', 'B')}))


def t_branches(tmp):
    print('1. boxing_branches')
    _rb, path = boxed_board(tmp, 'br')
    b = parse_kicad_pcb(path)
    pads = [p for fp in b.footprints.values() for p in fp.pads]
    s1 = next(p for p in pads if p.component_ref == 'S1')
    s2 = next(p for p in pads if p.component_ref == 'S2')
    br = opu.boxing_branches(b, s1, CLEARANCE, WIDTH)
    names = {b.nets[n].name: len(s) for n, s in br.items()}
    check('the B track boxes S1', names == {'B': 1}, str(names))
    far = opu.boxing_branches(b, s1, CLEARANCE, WIDTH,
                              frozen_ids={id(s) for s in b.segments})
    check('frozen copper is never a branch', far == {}, str(far))
    check('nothing boxes S2, 7 mm off',
          opu.boxing_branches(b, s2, CLEARANCE, WIDTH) == {})


def t_ships(tmp):
    print('2. a reroute that frees the pad ships')
    rb, path = boxed_board(tmp, 'ok')
    n0 = open_count(path)
    freed, calls, _b, _a = run(path, rb, [SIG_JOIN] + B_DETOUR)
    check('one pad freed', freed == 1 and n0 == 1 and open_count(path) == 0,
          f'freed {freed}, open {n0} -> {open_count(path)}')
    check('rerouted SIG and B, SIG first',
          bool(calls) and calls[0][0] == ['SIG', 'B'],
          str([c[0] for c in calls]))
    check('the reroute input has no boxing track',
          bool(calls) and 'box-track' not in calls[0][1])


def t_lost(tmp):
    print('3. a reroute that costs B a pad is refused')
    rb, path = boxed_board(tmp, 'lost')
    freed, calls, before, after = run(path, rb, [SIG_JOIN])
    check('refused, board unchanged', freed == 0 and before == after
          and len(calls) == 1, f'freed {freed}, calls {len(calls)}')


def t_drc(tmp):
    print('4. a reroute that adds a DRC finding is refused')
    rb, path = boxed_board(tmp, 'drc')
    freed, calls, before, after = run(path, rb, [SIG_JOIN] + B_SHORT)
    check('refused, board unchanged', freed == 0 and before == after
          and len(calls) == 1, f'freed {freed}, calls {len(calls)}')


def t_nothing(tmp):
    print('5. a reroute that writes no board')
    rb, path = boxed_board(tmp, 'none')
    freed, calls, before, after = run(path, rb, [], write=False)
    check('board unchanged', freed == 0 and before == after
          and len(calls) == 1)


def t_frozen(tmp):
    print('6. locked and --keep-input-copper copper stay')
    rb, path = boxed_board(tmp, 'lock', locked=True)
    freed, calls, before, after = run(path, rb, [SIG_JOIN] + B_DETOUR)
    check('locked: never rerouted', freed == 0 and not calls
          and before == after)
    rb, path = boxed_board(tmp, 'keep')
    keep = {copper_item_key(s) for s in parse_kicad_pcb(path).segments}
    freed, calls, before, after = run(path, rb, [SIG_JOIN] + B_DETOUR,
                                      keep_input=keep)
    check('input copper: never rerouted', freed == 0 and not calls
          and before == after)


def route_cli(tmp, tag, rb, *extra):
    src = os.path.join(tmp, f'{tag}_in.kicad_pcb')
    out = os.path.join(tmp, f'{tag}_out.kicad_pcb')
    with open(src, 'w', encoding='utf-8') as fh:
        fh.write(rb.text())
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'py_router',
                                                     'route.py'),
                        src, out, '--nets', 'SIG', '--layers', 'F.Cu',
                        *extra],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    return r, r.stdout + r.stderr


def t_cli(tmp):
    print('7. CLI: COMPLETION:, --require-complete, --no-unbox')
    rb = RoutingBoard()
    rb.pad('S1', 10, 15, 'SIG')
    rb.pad('S2', 30, 15, 'SIG')
    r, out = route_cli(tmp, 'clean', rb, '--require-complete')
    check('a complete run exits 0',
          r.returncode == 0 and 'every routed net is complete' in out
          and 'Traceback' not in out, f'rc {r.returncode}')
    check('the pass ran', 'Open-pad unboxing (end of run)' in out)

    # S1 walled in by a LOCKED ring of B: nothing can reach it, and the pass
    # may not take locked copper up.
    rb = RoutingBoard()
    rb.pad('S1', 10, 15, 'SIG')
    rb.pad('S2', 30, 15, 'SIG')
    rb.pad('B1', 10, 12, 'B')
    rb.ring([(9.2, 14.2), (10.8, 14.2), (10.8, 15.8), (9.2, 15.8)], 'B',
            w=0.2, locked=True)
    rb.seg(10, 12, 10, 14.2, 'B', w=0.2, locked=True)
    r, out = route_cli(tmp, 'walled', rb, '--require-complete')
    check('a pad left open: exit 4, named',
          r.returncode == 4 and 'INCOMPLETE: 1 pad(s)' in out
          and 'COMPLETION: 1 pad(s) off their net: SIG@' in out
          and 'Traceback' not in out,
          f'rc {r.returncode}: {out.strip().splitlines()[-1:]}')
    r, out = route_cli(tmp, 'walled2', rb, '--no-unbox')
    check('--no-unbox: no pass, still graded, exit 0 without the gate',
          r.returncode == 0 and 'Open-pad unboxing' not in out
          and 'COMPLETION: 1 pad(s)' in out, f'rc {r.returncode}')


def main():
    tmp = tempfile.mkdtemp(prefix='krt_unbox_test_')
    try:
        for t in (t_branches, t_ships, t_lost, t_drc, t_nothing, t_frozen,
                  t_cli):
            t(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nFAILED: ' + ', '.join(FAILS) if FAILS else '\nALL PASS')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
