"""Tie a package's same-net pins to its exposed pad before routing.

A QFN/DFN pin of the exposed pad's net (GND on most parts) is joined to the
exposed pad by a short straight stub under the package, along the pin's
long axis. A board designer lays these first; the router left them to the
multipoint pass, which then found the strip between the pin row and the
exposed pad already taken by a signal routed under the part, or joined two
pins round the outside of the row and boxed in the pin between them.

The stub runs from the pin centre straight inward and ends ``depth`` inside
the exposed pad. It is laid only when it clears every other net's copper at
the run's clearance (the gates every pass that places new copper uses), and
only for axis-aligned pads on the exposed pad's copper layer whose centre
line meets the exposed pad's copper at least half a track width from its
corner.
"""
import math
from statistics import median

from kicad_parser import Segment

# An exposed pad is at least this many times the package's median SMD pad
# area, on a package with at least MIN_PINS SMD pads.
EP_AREA_FACTOR = 8.0
MIN_PINS = 8


def _smd(p):
    return (not getattr(p, 'drill', 0) and p.size_x > 0 and p.size_y > 0
            and any(L.endswith('.Cu') for L in p.layers))


def _axis_rect(p):
    return (not getattr(p, 'rect_rotation', 0)
            and not getattr(p, 'polygons', None))


def exposed_pads(pcb_data):
    """{component_ref: (exposed pads, the package's SMD pads, big-pad area)}
    for every package that has an exposed pad."""
    by_ref = {}
    for pads in pcb_data.pads_by_net.values():
        for p in pads:
            if _smd(p):
                by_ref.setdefault(p.component_ref, []).append(p)
    out = {}
    for ref, pads in by_ref.items():
        if len(pads) < MIN_PINS:
            continue
        big = EP_AREA_FACTOR * median(p.size_x * p.size_y for p in pads)
        eps = [p for p in pads if p.net_id and _axis_rect(p)
               and p.size_x * p.size_y >= big]
        if eps:
            out[ref] = (eps, pads, big)
    return out


def _stub_end(pin, ep, width, depth):
    """Where the pin's inward stub ends inside ``ep``, or None when the
    pin's centre line does not meet the exposed pad's copper."""
    hx, hy = ep.size_x / 2, ep.size_y / 2
    dx, dy = ep.global_x - pin.global_x, ep.global_y - pin.global_y
    if pin.size_y > pin.size_x:                       # long axis vertical
        if abs(dx) > hx - width / 2 or abs(dy) <= hy:
            return None
        return (pin.global_x, ep.global_y - math.copysign(hy - depth, dy))
    if pin.size_x > pin.size_y:                       # long axis horizontal
        if abs(dy) > hy - width / 2 or abs(dx) <= hx:
            return None
        return (ep.global_x - math.copysign(hx - depth, dx), pin.global_y)
    return None


def _already_tied(segments, pin, ep, layer):
    from check_drc import point_to_pad_distance
    for s in segments:
        if s.net_id != pin.net_id or s.layer != layer:
            continue
        for (ax, ay, bx, by) in ((s.start_x, s.start_y, s.end_x, s.end_y),
                                 (s.end_x, s.end_y, s.start_x, s.start_y)):
            if (point_to_pad_distance(ax, ay, pin) <= 0
                    and point_to_pad_distance(bx, by, ep) <= 0):
                return True
    return False


def plan_exposed_pad_ties(pcb_data, net_ids, width, clearance, clears,
                          depth=None):
    """The tie segments for the nets ``net_ids``: one per pin of an exposed
    pad's net whose stub passes ``clears(x1, y1, x2, y2, layer, net, w)``
    and keeps ``clearance`` from the other nets' ties planned before it.
    Pins already joined to their exposed pad by a track are skipped."""
    from geometry_utils import segment_to_segment_distance
    depth = width if depth is None else depth
    nets = set(net_ids)
    ties = []
    for ref, (eps, pads, big) in sorted(exposed_pads(pcb_data).items()):
        for ep in eps:
            if ep.net_id not in nets:
                continue
            ep_layers = [L for L in ep.layers if L.endswith('.Cu')]
            for pin in pads:
                if (pin.net_id != ep.net_id or not _axis_rect(pin)
                        or pin.size_x * pin.size_y >= big):
                    continue
                layer = next((L for L in pin.layers if L in ep_layers), None)
                end = _stub_end(pin, ep, width, depth) if layer else None
                if end is None or _already_tied(pcb_data.segments, pin, ep, layer):
                    continue
                x1, y1, (x2, y2) = pin.global_x, pin.global_y, end
                if not clears(x1, y1, x2, y2, layer, pin.net_id, width):
                    continue
                if any(t.layer == layer and t.net_id != pin.net_id and
                       segment_to_segment_distance(
                           x1, y1, x2, y2, t.start_x, t.start_y, t.end_x,
                           t.end_y) < clearance + (width + t.width) / 2
                       for t in ties):
                    continue
                ties.append(Segment(start_x=x1, start_y=y1, end_x=x2, end_y=y2,
                                    width=width, layer=layer, net_id=pin.net_id))
    return ties


def tie_pins_to_exposed_pads(pcb_data, net_ids, config):
    """Plan the ties at the run's clearance and track width and add them to
    ``pcb_data``. Returns the new segments."""
    from pcb_modification import _copper_clearance_gates
    width = getattr(config, 'track_width', 0) or 0.15
    clearance = getattr(config, 'clearance', 0.1)
    gates = _copper_clearance_gates(
        pcb_data, clearance, getattr(config, 'net_clearances', None) or None,
        getattr(config, 'board_edge_clearance', 0.0) or 0.0, config)
    ties = plan_exposed_pad_ties(pcb_data, net_ids, width, clearance,
                                 gates.clears)
    if ties:
        pcb_data.segments = list(pcb_data.segments) + ties
        for attr in ('_foreign_seg_arr_cache', '_foreign_via_arr_cache',
                     '_foreign_pad_arr_cache'):
            if getattr(pcb_data, attr, None) is not None:
                setattr(pcb_data, attr, None)
    return ties
