"""
routing.py
----------
Baseline routing + alternate-path generation.

Two jobs:
  1. shortest_path_routing()  -> the BASELINE every network uses today
     (OSPF-style shortest path, no congestion awareness at all).
  2. candidate_paths()        -> the set of alternate routes we are allowed
     to consider when we reroute one flow (Block 6 "candidate interventions").
"""

import networkx as nx


def shortest_path_routing(G, flows):
    """BASELINE 1: route every flow along its shortest path (fewest hops).

    This is what a normal router does. It knows nothing about load, so all
    the congestion in our experiments is created by this baseline - which is
    exactly the problem our project tries to fix with the smallest possible
    correction.

    Unroutable flows (no path at all, e.g. a partitioned graph) are mapped to
    None instead of crashing.
    """
    paths = {}
    for flow in flows:
        try:
            paths[flow.fid] = nx.shortest_path(G, flow.src, flow.dst, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths[flow.fid] = None
    return paths


def candidate_paths(G, flow, current_path, max_candidates=8):
    """List alternate routes for ONE flow, shortest first.

    networkx.shortest_simple_paths yields loop-free paths in increasing order
    of cost (Yen's algorithm). We take the first `max_candidates` of them and
    drop the one the flow is already using.

    Returns [] when the flow has no alternative at all - the "no alternate
    path" edge case the faculty asked us to handle.
    """
    try:
        generator = nx.shortest_simple_paths(G, flow.src, flow.dst, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    alternatives = []
    for i, path in enumerate(generator):
        if i >= max_candidates:
            break
        if path != current_path:
            alternatives.append(path)
    return alternatives


def path_capacity_ok(G, path, demand, loads, link_key_fn):
    """Optional hard feasibility check: would this path physically overflow?

    Returns True if every link on `path` can absorb `demand` without going
    over 100% of its capacity. We use this as a constraint so the prototype
    never 'fixes' one link by destroying another.
    """
    for u, v in zip(path[:-1], path[1:]):
        key = link_key_fn(u, v)
        if loads.get(key, 0.0) + demand > G[u][v]["capacity"]:
            return False
    return True


def path_str(path):
    """Pretty-print a path: ['R1','R3','R6'] -> 'R1 -> R3 -> R6'."""
    if not path:
        return "UNROUTABLE"
    return " -> ".join(path)
