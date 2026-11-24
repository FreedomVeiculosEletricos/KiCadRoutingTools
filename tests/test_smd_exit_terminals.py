#!/usr/bin/env python3
"""The router lays SMD pad exits itself (smd_exits.py): a pad terminal is
searched from / to its compliant straight exits' mouths, each with its
heading (grid_router source_headings / target_headings), so the route leaves
the pad straight out of an edge -- Altium's SMD Entry -- and keeps straight
for the run's SMD To Corner distance before its first bend.

  1. exit_mouths: one mouth per allowed edge, at least the SMD To Corner
     distance past the outline, anchored inside the pad; a blocked exit is
     left out; --smd-to-corner moves the mouths out; with corner exits (the
     default) one more per corner, at 45 degrees on the corner's bisector,
     and none with them off;
  2. route.py on a board whose pads sit diagonally apart: with the exits no
     smd-entry / smd-to-corner is left; with --no-smd-exit-terminals (the
     control) the route leaves through the corners.

    python3 tests/test_smd_exit_terminals.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path[:0] = [TESTS_DIR, os.path.join(ROOT, 'py_router')]

from synth import RoutingBoard                                  # noqa: E402
from kicad_parser import parse_kicad_pcb                        # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def _board(tmp, extra=()):
    b = RoutingBoard(w=30, h=30)
    b.pad('R1', 10, 10, 'N1', sx=0.6, sy=1.0)
    b.pad('R2', 14, 13, 'N1', sx=0.6, sy=1.0)
    b.pad('R3', 10, 20, 'N2', sx=1.0, sy=0.6)
    b.pad('R4', 13, 17, 'N2', sx=1.0, sy=0.6)
    for args in extra:
        b.seg(*args)
    p = os.path.join(tmp, 'in.kicad_pcb')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(b.text())
    return p


def _parse(p):
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_kicad_pcb(p)


def _mouths(pcb, net=1, ref='R1'):
    from routing_config import GridRouteConfig
    import smd_exits
    cfg = GridRouteConfig(track_width=0.2, clearance=0.15)
    pad = next(p for p in pcb.pads_by_net[net] if p.component_ref == ref)
    return pad, smd_exits.exit_mouths(pad, 'F.Cu', 0, 0.2, cfg.grid_step, pcb,
                                      net, cfg, None)


def t_mouths(tmp):
    print("1. exit_mouths")
    import pad_entry
    pcb = _parse(_board(tmp))
    pad, ms_all = _mouths(pcb)
    ms = [m for m in ms_all if not (m[5] and m[6])]      # the edge exits
    heads = sorted({(m[5], m[6]) for m in ms})
    check("every edge of a 0.6 x 1.0 pad has an exit",
          heads == [(-1, 0), (0, -1), (0, 1), (1, 0)], heads)
    # strict exits run on every grid lane the side's flat part carries; the
    # centred lane is always one of them
    centred = {(m[5], m[6]) for m in ms
               if (m[4] if m[5] else m[3]) == (pad.global_y if m[5] else pad.global_x)}
    check("each edge keeps its centred lane", centred == set(heads), sorted(centred))
    ok = True
    for gx, gy, _li, ax, ay, sx, sy, _g in ms:
        mx, my = gx * 0.1, gy * 0.1
        half = (pad.size_x if sx else pad.size_y) / 2.0
        past = abs((mx - pad.global_x) if sx else (my - pad.global_y)) - half
        # anchored at the centre along the exit, on its lane across it
        ok &= past >= 0.25 - 1e-6 and (ax == pad.global_x if sx else ay == pad.global_y)
    check("every mouth >= 0.25 mm past the outline, anchored at the centre "
          "along its exit", ok)
    prev = pad_entry.set_smd_to_corner(0.6)
    try:
        _p, far = _mouths(pcb)
    finally:
        pad_entry.set_smd_to_corner(prev)
    check("--smd-to-corner 0.6 moves every mouth out",
          all(abs(f[0] - 100) + abs(f[1] - 100) > abs(m[0] - 100) + abs(m[1] - 100)
              for f, m in zip(sorted(far), sorted(ms))))
    t_corner_mouths(pad, ms_all, pcb)
    # foreign copper across R1's +x exit
    pcb = _parse(_board(tmp, extra=[(10.5, 9.0, 10.5, 11.0, 'N9', 0.15)]))
    _p, ms = _mouths(pcb)
    ms = [m for m in ms if not (m[5] and m[6])]
    check("an exit crossed by foreign copper is left out",
          sorted({(m[5], m[6]) for m in ms}) == [(-1, 0), (0, -1), (0, 1)],
          sorted({(m[5], m[6]) for m in ms}))


def t_corner_mouths(pad, ms_all, pcb):
    import math
    import pad_entry
    from pad_entry import exit_crossing, corner_exit_ok, grade_pad_exit
    cs = [m for m in ms_all if m[5] and m[6]]
    check("every corner of a 0.6 x 1.0 pad has a 45-degree exit",
          sorted({(m[5], m[6]) for m in cs}) == [(-1, -1), (-1, 1), (1, -1), (1, 1)],
          sorted({(m[5], m[6]) for m in cs}))
    ok = True
    for gx, gy, _li, ax, ay, sx, sy, _g in cs:
        mx, my = gx * 0.1, gy * 0.1
        ok &= abs(abs(mx - ax) - abs(my - ay)) < 1e-6        # exactly 45 degrees
        cr = exit_crossing(pad, [(ax, ay), (mx, my)])
        ok &= cr is not None and corner_exit_ok(pad, *cr[1], mx - ax, my - ay)
        ok &= math.hypot(mx - cr[1][0], my - cr[1][1]) >= 0.25 - 1e-6
        ok &= not grade_pad_exit(pad, [(ax, ay), (mx, my), (mx + sx * 0.1, my)], 0.2)
        ok &= pad_entry.pad_depth(pad, ax, ay) >= pad_entry.required_depth(pad, 0.2) - 0.002
    check("each corner exit runs through its corner on the bisector, full depth, "
          ">= 0.25 mm straight past the outline", ok)
    from types import SimpleNamespace
    from pad_entry import strict_exit_faults
    legs = [SimpleNamespace(start_x=ax, start_y=ay, end_x=gx * 0.1, end_y=gy * 0.1,
                            width=0.2, layer='F.Cu', net_id=1)
            for gx, gy, _li, ax, ay, sx, sy, _g in cs]
    bad = strict_exit_faults(legs, [], [pad])
    check("the strict grade passes every corner leg", not bad, dict(bad))
    tilt = [SimpleNamespace(start_x=pad.global_x, start_y=pad.global_y,
                            end_x=pad.global_x + 1.0, end_y=pad.global_y + 1.0,
                            width=0.2, layer='F.Cu', net_id=1)]
    check("a 45-degree leg off the corner is a strict diag-exit",
          any(k[2] == 'diag-exit' for k in strict_exit_faults(tilt, [], [pad])))
    prev = pad_entry.set_corner_exits(False)
    try:
        import smd_exits
        from routing_config import GridRouteConfig
        cfg = GridRouteConfig(track_width=0.2, clearance=0.15)
        off = smd_exits.exit_mouths(pad, 'F.Cu', 0, 0.2, cfg.grid_step, pcb, 1, cfg, None)
    finally:
        pad_entry.set_corner_exits(prev)
    check("corner exits off: edge exits only", all(not (m[5] and m[6]) for m in off),
          sorted({(m[5], m[6]) for m in off}))


def _route(tmp, name, extra):
    inp = _board(tmp)
    out = os.path.join(tmp, name + '.kicad_pcb')
    r = subprocess.run([sys.executable, '-X', 'utf8',
                        os.path.join(ROOT, 'py_router', 'route.py'), inp,
                        '--output', out, '--track-width', '0.2',
                        '--clearance', '0.15'] + extra,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.isfile(out):
        return None
    from check_weird import check_weird
    pcb = _parse(out)
    with contextlib.redirect_stdout(io.StringIO()):
        fs, _ = check_weird(pcb, tolerance=0)
    return pcb, [f for f in fs if f['category'] in ('smd-entry', 'smd-to-corner')]


def t_route(tmp):
    print("2. route.py lays the exits")
    got = _route(tmp, 'on', [])
    check("the board routes", got is not None)
    if got:
        pcb, bad = got
        check("no smd-entry / smd-to-corner left", not bad,
              '; '.join(f"{f['category']} {f['detail']}" for f in bad))
        check("both nets routed", {s.net_id for s in pcb.segments} >= {1, 2})
    ctl = _route(tmp, 'off', ['--no-smd-exit-terminals'])
    check("control: without the exits the route leaves through a corner",
          ctl is not None and any(f['category'] == 'smd-entry' for f in ctl[1]),
          ctl and [f['category'] for f in ctl[1]])


def main():
    with tempfile.TemporaryDirectory(prefix='smdx_') as tmp:
        t_mouths(tmp)
        t_route(tmp)
    if FAILS:
        print(f"\nFAILED {len(FAILS)}: " + "; ".join(FAILS))
        sys.exit(1)
    print("\nALL PASS")


if __name__ == '__main__':
    main()
