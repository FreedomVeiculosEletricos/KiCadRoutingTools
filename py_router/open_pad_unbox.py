"""End of a CLI run: free the pads the run left open because other nets'
copper boxes them in, and reroute.

A pad stays open when the copper the run laid round it leaves no channel a
track fits through: a power net's tap run along a 0402 column between the
pad and the board, a ground track wrapped round a decoupling pad on its way
between two other pads. The rip-up ladder cannot free it -- the plane repair
never rips a plane net, and the reconciliation does not rip copper the run
itself routed -- although the boxing net could take another path or drop a
via of its own.

For each open pad, the BRANCHES of other nets that box it are taken off a
copy of the shipped board: every track that passes too close to the pad for
a track to fit between it and the pad, extended along its net to the next
pad, via or junction. The open pad's net and the branches' nets are then
routed again on that copy by the run's own engine (``route_fn``) with
everything else in place. The copy ships only when, measured on the whole
board, it has fewer pads open, no net it touched lost a pad, and no more DRC
findings of any kind than the board it replaces. Otherwise the shipped board
is left as it was. Rounds repeat while a round frees a pad, each open pad
tried once.
"""
import os
import shutil
import tempfile
from collections import Counter

# A round frees at most one pad and then re-reads the board; rounds repeat
# while one frees a pad. Every pad is tried once, so the pass ends on its
# own; MAX_ROUNDS only bounds a board with very many open pads.
MAX_ROUNDS = 64
MAX_BRANCH_SEGMENTS = 40


def _pad_copper_layers(pad):
    return [L for L in pad.layers if L.endswith('.Cu')] if not \
        any(L == '*.Cu' for L in pad.layers) else None


def open_pads(board, scope_ids):
    """[(net_id, (x, y))] of the pads the board leaves off their net."""
    from improvement_gate import grade_nets
    out = []
    for nid, g in sorted(grade_nets(board, sorted(scope_ids)).items()):
        for fp in g['failed_pads']:
            out.append((nid, (fp['x'], fp['y'])))
    return out


def _pad_at(board, nid, xy):
    best = None
    for p in board.pads_by_net.get(nid) or []:
        d = abs(p.global_x - xy[0]) + abs(p.global_y - xy[1])
        if best is None or d < best[0]:
            best = (d, p)
    return best[1] if best and best[0] < 0.05 else None


def _seg_pad_distance(s, pad, step=0.02):
    """Centre line of segment ``s`` to ``pad``'s copper (0 inside)."""
    import math
    from check_drc import point_to_pad_distance
    dx, dy = s.end_x - s.start_x, s.end_y - s.start_y
    n = max(1, int(math.hypot(dx, dy) / step))
    return min(point_to_pad_distance(s.start_x + dx * i / n,
                                     s.start_y + dy * i / n, pad)
               for i in range(n + 1))


def boxing_branches(board, pad, clearance, width, frozen_ids=()):
    """{net_id: [segments]}: the other nets' branches that pass within
    2 * clearance + width of ``pad``'s copper on one of its layers (no track
    of ``width`` fits between), each extended along its net through plain
    two-track joints to the next pad, via or junction."""
    from check_drc import point_to_pad_distance
    layers = _pad_copper_layers(pad)
    reach = 2 * clearance + width
    frozen = set(frozen_ids)
    near = [s for s in board.segments
            if s.net_id and s.net_id != pad.net_id and id(s) not in frozen
            and (layers is None or s.layer in layers)
            and _seg_pad_distance(s, pad) - s.width / 2 < reach]
    if not near:
        return {}
    pads_of = {}
    for nid in {s.net_id for s in near}:
        pads_of[nid] = board.pads_by_net.get(nid) or []

    def rk(x, y):
        return (round(x, 4), round(y, 4))
    out = {}
    for nid in sorted({s.net_id for s in near}):
        segs = [s for s in board.segments if s.net_id == nid]
        ends = Counter()
        by_end = {}
        for s in segs:
            for e in (rk(s.start_x, s.start_y), rk(s.end_x, s.end_y)):
                ends[(s.layer, e)] += 1
                by_end.setdefault((s.layer, e), []).append(s)
        vias = {rk(v.x, v.y) for v in board.vias if v.net_id == nid}

        def stop(layer, e):
            if ends[(layer, e)] != 2 or e in vias:
                return True
            return any(point_to_pad_distance(e[0], e[1], p) <= 0
                       for p in pads_of[nid] if layer in p.layers
                       or '*.Cu' in p.layers)
        branch = []
        seen = set()
        todo = [s for s in near if s.net_id == nid]
        while todo:
            s = todo.pop()
            if id(s) in seen:
                continue
            if id(s) in frozen or len(branch) >= MAX_BRANCH_SEGMENTS:
                return {}
            seen.add(id(s))
            branch.append(s)
            for e in (rk(s.start_x, s.start_y), rk(s.end_x, s.end_y)):
                if stop(s.layer, e):
                    continue
                todo.extend(t for t in by_end[(s.layer, e)] if id(t) not in seen)
        out[nid] = branch
    return out


def _drc_counts(path, clearance, hole_to_hole, edge):
    import contextlib
    import io
    from check_drc import run_drc
    from kicad_parser import parse_kicad_pcb
    from list_nets import net_clearance_map
    pcb = parse_kicad_pcb(path)
    ncl = net_clearance_map(path, [n.name for n in pcb.nets.values()]) or None
    with contextlib.redirect_stdout(io.StringIO()):
        rows = run_drc(path, clearance=clearance, quiet=True, max_print=0,
                       check_sizes=False, pcb_data=pcb, net_clearances=ncl,
                       hole_to_hole_clearance=hole_to_hole,
                       board_edge_clearance=edge, print_summary=False)
    return Counter(r.get('type') for r in rows)


def _no_worse(before, after):
    return all(after[k] <= before[k] for k in after)


def unbox_open_pads(output_file, scope_names, route_fn, clearance, width,
                    hole_to_hole=0.2, edge=0.0, keep_input=None):
    """Run the pass on the written board ``output_file`` in place.
    ``route_fn(input_file, output_file, net_names)`` routes the named nets of
    a board with the run's own options. ``keep_input`` (a set of copper
    signature keys) marks input copper the run may not move. Returns the
    number of pads freed."""
    from kicad_parser import parse_kicad_pcb
    from improvement_gate import grade_nets, copper_item_key
    freed_total = 0
    tried = set()
    for _round in range(MAX_ROUNDS):
        board = parse_kicad_pcb(output_file)
        name_of = {nid: n.name for nid, n in board.nets.items()}
        scope = {nid for nid, nm in name_of.items() if nm in set(scope_names)}
        opens = open_pads(board, scope)
        if not opens:
            break
        before_grade = grade_nets(board, sorted(scope))
        before_open = sum(len(g['failed_pads']) for g in before_grade.values())
        before_drc = None
        frozen = {id(s) for s in board.segments
                  if getattr(s, 'locked', False)
                  or (keep_input and copper_item_key(s) in keep_input)}
        progressed = False
        for nid, xy in opens:
            key = (name_of[nid], round(xy[0], 3), round(xy[1], 3))
            if key in tried:
                continue
            tried.add(key)
            pad = _pad_at(board, nid, xy)
            if pad is None:
                continue
            branches = boxing_branches(board, pad, clearance, width, frozen)
            branches = {b: s for b, s in branches.items() if b in scope}
            if not branches:
                continue
            nets = [name_of[nid]] + [name_of[b] for b in branches]
            removed = [s for segs in branches.values() for s in segs]
            label = f"{pad.component_ref}.{pad.pad_number} ({name_of[nid]})"
            print(f"  Open pad {label}: boxed in by "
                  + ", ".join(f"{name_of[b]} ({len(s)} track(s))"
                              for b, s in branches.items())
                  + " -- rerouting " + ", ".join(nets) + " without them")
            work = tempfile.mkdtemp(prefix='krt_unbox_')
            try:
                tin = os.path.join(work, 'in.kicad_pcb')
                tout = os.path.join(work, 'out.kicad_pcb')
                shutil.copyfile(output_file, tin)
                base = os.path.splitext(output_file)[0]
                for ext in ('.kicad_pro', '.kicad_dru'):
                    if os.path.isfile(base + ext):
                        shutil.copyfile(base + ext, os.path.join(work, 'in' + ext))
                from route import _strip_written_copper
                n_s, _ = _strip_written_copper(tin, removed, [], name_of)
                if n_s != len(removed):
                    print(f"    left as routed: found {n_s} of the "
                          f"{len(removed)} track(s) in the file")
                    continue
                try:
                    route_fn(tin, tout, nets)
                except SystemExit:
                    pass
                if not os.path.isfile(tout):
                    print("    left as routed: the reroute wrote no board")
                    continue
                after = parse_kicad_pcb(tout)
                after_grade = grade_nets(after, sorted(scope))
                after_open = sum(len(g['failed_pads'])
                                 for g in after_grade.values())
                lost = [name_of[n] for n, g in after_grade.items()
                        if len(g['failed_pads'])
                        > len((before_grade.get(n) or {}).get('failed_pads') or [])]
                if after_open >= before_open or lost:
                    print(f"    left as routed: open pads {before_open} -> "
                          f"{after_open}"
                          + (f", {', '.join(lost)} lost pads" if lost else ""))
                    continue
                if before_drc is None:
                    before_drc = _drc_counts(output_file, clearance,
                                             hole_to_hole, edge)
                after_drc = _drc_counts(tout, clearance, hole_to_hole, edge)
                if not _no_worse(before_drc, after_drc):
                    worse = {k: (before_drc[k], v) for k, v in after_drc.items()
                             if v > before_drc[k]}
                    print(f"    left as routed: DRC worse {worse}")
                    continue
                shutil.copyfile(tout, output_file)
                print(f"    freed: open pads {before_open} -> {after_open}")
                freed_total += before_open - after_open
                progressed = True
                break           # re-read the board before the next pad
            finally:
                shutil.rmtree(work, ignore_errors=True)
        if not progressed:
            break
    return freed_total
