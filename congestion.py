"""
congestion.py
-------------
Architecture Block 2 (part 2): flow -> link mapping, link loads,
utilisation, congestion detection, and the global congestion metric C(G).

THE CONGESTION METRIC
=====================
We use "total excess utilisation above a threshold":

        u_e  = load_e / capacity_e                (utilisation of link e)
        C(G) = SUM over links e of  max(0, u_e - THETA)

with THETA = 0.80 (a link is "congested" once it is more than 80% full).

Why this metric?
  1. C(G) = 0 exactly when no link is congested. A clean "healthy network"
     baseline.
  2. It is MONOTONE: removing traffic can never increase any link load,
     so C can never go up. This guarantees that our counterfactual score
     DeltaC_i = C(G) - C(G without flow i) is always >= 0, i.e. no flow can
     ever get a negative amount of blame. (See README, "Does the formula
     behave correctly?")
  3. It counts ALL congested links, not only the worst one, so an
     intervention that fixes link A while breaking link B is correctly
     scored as "no real improvement".
  4. It ignores links below the threshold, so a flow using only quiet links
     correctly gets zero responsibility.

We also report MLU (Maximum Link Utilisation) = max_e u_e as a second,
very standard metric, so results can be sanity-checked against the usual
traffic-engineering literature.
"""

from network import link_key

# A link is declared congested when it is more than 80% full.
CONGESTION_THRESHOLD = 0.80


def compute_link_loads(G, flows, paths):
    """Add up how much traffic every link carries.

    paths : dict  flow_id -> list of nodes, e.g. {"F1": ["R1","R3","R6","R8"]}

    For every flow we walk along its path and add its demand to each link
    on that path. This is the "flow-link interaction" of Block 2.
    """
    loads = {link_key(u, v): 0.0 for u, v in G.edges()}

    for flow in flows:
        path = paths.get(flow.fid)
        if not path:            # unroutable flow (no path exists) -> carries nothing
            continue
        for u, v in zip(path[:-1], path[1:]):
            loads[link_key(u, v)] += flow.demand

    return loads


def compute_utilisation(G, loads):
    """Convert absolute loads (Mbps) into utilisation (a fraction of capacity)."""
    util = {}
    for (u, v), load in loads.items():
        util[(u, v)] = load / G[u][v]["capacity"]
    return util


def congested_links(util, threshold=CONGESTION_THRESHOLD):
    """Return the congested links, worst first.

    Returns a list of (link, utilisation) sorted by utilisation descending.
    """
    bad = [(link, u) for link, u in util.items() if u > threshold]
    bad.sort(key=lambda item: item[1], reverse=True)
    return bad


def congestion_metric(util, threshold=CONGESTION_THRESHOLD):
    """C(G) = sum of excess utilisation above the threshold.

    Returns 0.0 for a completely healthy network.
    """
    return sum(max(0.0, u - threshold) for u in util.values())


def max_link_utilisation(util):
    """MLU = the utilisation of the single busiest link."""
    return max(util.values()) if util else 0.0


def network_state(G, flows, paths, threshold=CONGESTION_THRESHOLD):
    """Compute everything about one network state in a single call.

    Returns a dictionary; this dict is what we compare BEFORE vs AFTER.
    """
    loads = compute_link_loads(G, flows, paths)
    util = compute_utilisation(G, loads)
    return {
        "loads": loads,
        "util": util,
        "C": congestion_metric(util, threshold),
        "MLU": max_link_utilisation(util),
        "congested": congested_links(util, threshold),
        "threshold": threshold,
    }


def flows_on_link(flows, paths, link):
    """Which flows actually traverse a given link? (needed for Block 4)"""
    result = []
    for flow in flows:
        path = paths.get(flow.fid)
        if not path:
            continue
        links = {link_key(u, v) for u, v in zip(path[:-1], path[1:])}
        if link in links:
            result.append(flow)
    return result
