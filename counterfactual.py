"""
counterfactual.py
-----------------
Architecture Block 3: "Counterfactual Removal Operator".

THE ONE IDEA IN THIS FILE
=========================
For each flow f_i we ask a what-if question:

    "If flow f_i had never existed, how congested would the network be?"

We answer it by literally deleting the flow from the list, recomputing the
link loads, and recomputing C. That gives the counterfactual state G_i^-1
and its congestion C(G_i^-1).

The difference

        DeltaC_i = C(G) - C(G_i^-1)

is how much congestion disappears when f_i disappears.

HONEST WORDING (important for the viva)
=======================================
This is *counterfactual attribution inside a simulation*, not a claim of
real-world causal certainty. It is valid here only because we control the
simulator: we can actually re-run the network without the flow. In a real
deployment you cannot delete a customer's traffic to see what happens, so
you would need an estimated model - and then the causal claim becomes much
weaker. We say this explicitly rather than over-claiming.
"""

from congestion import network_state


def remove_flow(flows, fid):
    """Return a copy of the flow list with flow `fid` deleted."""
    return [f for f in flows if f.fid != fid]


def counterfactual_states(G, flows, paths, threshold):
    """Run the removal test for EVERY flow.

    Returns dict: flow_id -> {
        "C_without" : congestion after removing this flow,
        "delta"     : DeltaC_i = C(G) - C_without   (>= 0 always),
        "MLU_without": max link utilisation after removal,
    }

    Complexity: O(F * (F * P + E)) - one full recomputation per flow.
    With 5-10 flows this is instant. For thousands of flows you would use
    an incremental update instead; noted as future work.
    """
    factual = network_state(G, flows, paths, threshold)
    C_factual = factual["C"]

    results = {}
    for flow in flows:
        reduced_flows = remove_flow(flows, flow.fid)
        hypothetical = network_state(G, reduced_flows, paths, threshold)

        delta = C_factual - hypothetical["C"]
        # Guard against tiny negative values from floating-point noise.
        # A true negative is mathematically impossible for our metric.
        if -1e-12 < delta < 0:
            delta = 0.0

        results[flow.fid] = {
            "C_without": hypothetical["C"],
            "delta": delta,
            "MLU_without": hypothetical["MLU"],
        }

    return C_factual, results


def counterfactual_on_link(G, flows, paths, threshold, link):
    """Per-link version: how much does each flow contribute to ONE link?

    Used when there are several congested links and we want to explain a
    specific one. Returns flow_id -> excess utilisation removed from `link`.
    """
    from congestion import compute_link_loads, compute_utilisation

    base_util = compute_utilisation(G, compute_link_loads(G, flows, paths))[link]
    base_excess = max(0.0, base_util - threshold)

    contributions = {}
    for flow in flows:
        reduced = remove_flow(flows, flow.fid)
        util = compute_utilisation(G, compute_link_loads(G, reduced, paths))[link]
        contributions[flow.fid] = base_excess - max(0.0, util - threshold)
    return contributions
