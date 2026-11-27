#!/usr/bin/env python3
"""Apply gives live board items the routed result's uuids on KiCad 10.

Some KiCad 10 bindings have no EDA_ITEM.SetUuid and expose m_Uuid
read-only, so every Apply that laid a track or via with a known uuid raised
AttributeError and left the board half applied. result_apply._set_uuid keeps
SetUuid where the bindings have it and otherwise copies the uuid into the
item's own KIID with KIID.Clone.

  1. (no KiCad) an item WITH SetUuid gets it called; one WITHOUT gets its
     m_Uuid cloned in place, and neither path touches the other;
  2. (KiCad's python) on a real PCB_TRACK and PCB_VIA, the Clone path --
     forced through a proxy that hides SetUuid -- sets the uuid, the board
     saves it, and a reload reads it back; the SetUuid path does the same.
     Skips (77) without KiCad's bundled python.

    python3 tests/test_result_apply_set_uuid.py
"""
import json
import os
import subprocess
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(ROOT, 'py_router'))
sys.path.insert(0, ROOT)

from kicad_exact_fill import find_kicad_python  # noqa: E402

RUN_ALL_TIMEOUT = 300
SKIP_EXIT = 77
FAILS = []


def check(name, cond, detail=''):
    if not cond:
        FAILS.append(name)
    print(('  PASS ' if cond else '  FAIL ') + name
          + (f'  {detail}' if detail else ''))


def _set_uuid():
    # result_apply imports nothing from wx or pcbnew at module level.
    from kicad_routing_plugin.result_apply import _set_uuid as f
    return f


class FakeKIID:
    def __init__(self, s=''):
        self.s = s

    def Clone(self, other):
        self.s = other.s


class FakePcbnew:
    KIID = FakeKIID


class WithSet:
    def __init__(self):
        self.set_to = None
        self.m_Uuid = FakeKIID('old')

    def SetUuid(self, k):
        self.set_to = k.s


class WithoutSet:
    def __init__(self):
        self.m_Uuid = FakeKIID('old')


def unit():
    print('1. both paths, no KiCad')
    f = _set_uuid()
    a, b = WithSet(), WithoutSet()
    f(a, 'u-1', FakePcbnew)
    f(b, 'u-2', FakePcbnew)
    check('SetUuid used where the bindings have it',
          a.set_to == 'u-1' and a.m_Uuid.s == 'old', f'{a.set_to} {a.m_Uuid.s}')
    check('m_Uuid cloned in place where they do not', b.m_Uuid.s == 'u-2',
          b.m_Uuid.s)


PROBE = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import pcbnew
from kicad_routing_plugin.result_apply import _set_uuid

class NoSet:
    """The item as KiCad 10 bindings without SetUuid expose it."""
    def __init__(self, item):
        object.__setattr__(self, '_i', item)
    def __getattr__(self, n):
        if n == 'SetUuid':
            raise AttributeError(n)
        return getattr(self._i, n)

out = sys.argv[2]
b = pcbnew.BOARD()
want = {}
for i, (kind, proxy) in enumerate([('track', True), ('via', True),
                                   ('track', False), ('via', False)]):
    it = pcbnew.PCB_TRACK(b) if kind == 'track' else pcbnew.PCB_VIA(b)
    if kind == 'track':
        it.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(1 + i), pcbnew.FromMM(1)))
        it.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(1 + i), pcbnew.FromMM(5)))
        it.SetWidth(pcbnew.FromMM(0.2))
    else:
        it.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(10 + 2 * i), pcbnew.FromMM(3)))
    u = '1235%04d-0000-4000-8000-0000000000%02d' % (i, i)
    target = NoSet(it) if proxy else it
    if proxy:
        assert not hasattr(target, 'SetUuid')
    _set_uuid(target, u, pcbnew)
    want[u] = {'kind': kind, 'clone': proxy,
               'live': it.m_Uuid.AsString()}
    b.Add(it)
pcbnew.SaveBoard(out, b)
got = {t.m_Uuid.AsString() for t in pcbnew.LoadBoard(out).GetTracks()}
print('RESULT ' + json.dumps({'want': want, 'reloaded': sorted(got),
                              'has_setuuid': hasattr(pcbnew.PCB_TRACK(b), 'SetUuid')}))
'''


def live():
    print('2. real pcbnew: set, save, reload')
    kpy = find_kicad_python()
    if kpy is None:
        print("SKIP: KiCad's bundled python not found")
        return False
    d = tempfile.mkdtemp(prefix='t_setuuid_')
    out = os.path.join(d, 'b.kicad_pcb')
    r = subprocess.run([kpy, '-c', PROBE, ROOT, out], capture_output=True,
                       text=True, timeout=240)
    rows = [json.loads(ln[len('RESULT '):]) for ln in r.stdout.splitlines()
            if ln.startswith('RESULT ')]
    if r.returncode != 0 or len(rows) != 1:
        print(f'BROKEN TEST: the pcbnew probe exited {r.returncode}:\n'
              f'{r.stdout[-2000:]}\n{r.stderr[-3000:]}')
        FAILS.append('probe')
        return True
    row = rows[0]
    print(f"  (these bindings {'have' if row['has_setuuid'] else 'lack'} SetUuid)")
    for u, w in sorted(row['want'].items()):
        how = 'Clone' if w['clone'] else 'SetUuid/native'
        check(f"{w['kind']} via {how}: live uuid set", w['live'] == u, w['live'])
        check(f"{w['kind']} via {how}: survives save + reload",
              u in row['reloaded'], str(row['reloaded']))
    return True


def main():
    unit()
    ran = live()
    print('\nFAILED: ' + ', '.join(FAILS) if FAILS else '\nALL PASS')
    if FAILS:
        return 1
    return 0 if ran else SKIP_EXIT


if __name__ == '__main__':
    sys.exit(main())
