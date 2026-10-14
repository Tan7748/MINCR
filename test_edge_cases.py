"""
test_edge_cases.py
------------------
Proves that the prototype handles the five awkward situations the faculty
asked about, instead of crashing or silently producing nonsense.

Run with:   python test_edge_cases.py

Each test prints what happened, and asserts the expected behaviour, so a
failure would stop the script with an AssertionError.
"""

import networkx as nx

from network import build_topology, NODE_POSITIONS
from traffic import Flow
from congestion import network_state, CONGESTION_THRESHOLD
from counterfactual import counterfactual_states
from responsibility import responsibility_scores, rank_flows, tied_group, any_responsibility
from routing import shortest_path_routing, candidate_paths
from main import run_pipeline

TH = CONGESTION_THRESHOLD


def banner(text):
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


# ---------------------------------------------------------------------------
def test_no_congestion():
    banner("CASE 1: no congestion exists")
    G = build_topology()
    flows = [Flow("F1", "R1", "R8", 5.0), Flow("F2", "R2", "R7", 4.0)]
    result = run_pipeline(G, flows, TH, verbose=False)

    print(f"C(G) = {result['state_before']['C']:.4f}, "
          f"status = {result['status']}, flows moved = {len(result['log'])}")
    assert result["state_before"]["C"] == 0.0
    assert result["status"] == "healthy"
    assert result["log"] == []
    print("PASS: the system correctly does nothing.")


# ---------------------------------------------------------------------------
def test_removal_does_not_help():
    banner("CASE 2: removing a single flow does not remove the congestion")
    # Four equal flows share one thin link. The link is so overloaded that
    # deleting any ONE of them still leaves it above the threshold, so every
    # DeltaC is positive but none is sufficient on its own.
    G = build_topology()
    flows = [Flow(f"F{i}", "R2", "R7", 30.0) for i in range(1, 5)]
    paths = shortest_path_routing(G, flows)
    C, deltas = counterfactual_states(G, flows, paths, TH)
    scores = responsibility_scores(deltas, flows)

    print(f"C(G) = {C:.4f}")
    for fid in rank_flows(scores):
        print(f"   {fid}: C without = {deltas[fid]['C_without']:.4f}, "
              f"DeltaC = {scores[fid]['delta']:.4f}, CR = {scores[fid]['CR']:.3f}")
    assert all(d["C_without"] > 0 for d in deltas.values()), \
        "removing one flow should NOT clear the congestion here"
    assert all(s["delta"] >= 0 for s in scores.values()), "DeltaC must never be negative"
    print("PASS: DeltaC stays >= 0 and the residual congestion is reported honestly.")


# ---------------------------------------------------------------------------
def test_equal_responsibility():
    banner("CASE 3: several flows have exactly equal responsibility")
    G = build_topology()
    flows = [Flow("F1", "R2", "R7", 21.0), Flow("F2", "R2", "R7", 21.0)]
    paths = shortest_path_routing(G, flows)
    C, deltas = counterfactual_states(G, flows, paths, TH)
    scores = responsibility_scores(deltas, flows)
    ranked = rank_flows(scores)
    tied = tied_group(scores, ranked)

    print(f"C(G) = {C:.4f}, CR = "
          + ", ".join(f"{f}:{scores[f]['CR']:.3f}" for f in ranked))
    print(f"tied group = {tied}  ->  deterministic pick = {ranked[0]}")
    assert len(tied) == 2
    assert ranked[0] == "F1"      # equal CR, equal demand -> alphabetical
    print("PASS: ties are detected and broken by a documented, repeatable rule.")


# ---------------------------------------------------------------------------
def test_no_alternate_path():
    banner("CASE 4: a flow has no alternate path at all")
    # A tiny chain topology: R1 - R2 - R3. There is exactly one route.
    G = nx.Graph()
    G.add_edge("R1", "R2", capacity=10.0, weight=1.0)
    G.add_edge("R2", "R3", capacity=10.0, weight=1.0)
    for n in G.nodes():
        G.nodes[n]["pos"] = NODE_POSITIONS.get(n, (0.0, 0.0))

    flow = Flow("F1", "R1", "R3", 9.5)
    paths = shortest_path_routing(G, [flow])
    options = candidate_paths(G, flow, paths["F1"])
    state = network_state(G, [flow], paths, TH)

    print(f"only path = {paths['F1']}, alternates found = {options}")
    print(f"C(G) = {state['C']:.4f} (link is 95% full)")
    result = run_pipeline(G, [flow], TH, verbose=False)
    print(f"status = {result['status']}, flows moved = {len(result['log'])}")
    assert options == []
    assert result["status"] == "no_feasible_action"
    print("PASS: the engine reports 'no feasible action' instead of crashing.")


# ---------------------------------------------------------------------------
def test_multiple_congested_links():
    banner("CASE 5: several links are congested at the same time")
    G = build_topology()
    flows = [
        Flow("F1", "R2", "R7", 22.0),
        Flow("F2", "R2", "R7", 21.0),
        Flow("F3", "R1", "R5", 40.0),
        Flow("F4", "R1", "R5", 35.0),
    ]
    paths = shortest_path_routing(G, flows)
    state = network_state(G, flows, paths, TH)
    print(f"congested links = "
          + ", ".join(f"{a}--{b} ({u * 100:.1f}%)" for (a, b), u in state["congested"]))
    print(f"C(G) = {state['C']:.4f} (sums the excess of BOTH links)")
    result = run_pipeline(G, flows, TH, verbose=False)
    print(f"after intervention: C(G) = {result['state_after']['C']:.4f}, "
          f"congested links = {len(result['state_after']['congested'])}, "
          f"flows moved = {len(result['log'])}, status = {result['status']}")
    assert len(state["congested"]) >= 2
    assert result["state_after"]["C"] <= state["C"]
    print("PASS: C(G) accounts for all congested links; congestion never increases.")


# ---------------------------------------------------------------------------
def test_monotonicity():
    banner("CASE 6 (sanity check): DeltaC_i is never negative")
    G = build_topology()
    from traffic import random_flows
    for seed in range(1, 21):
        flows = random_flows(G, 8, seed=seed, min_demand=10, max_demand=45)
        paths = shortest_path_routing(G, flows)
        _, deltas = counterfactual_states(G, flows, paths, TH)
        for fid, info in deltas.items():
            assert info["delta"] >= 0.0, f"negative delta for {fid} at seed {seed}"
    print("Checked 20 random scenarios x 8 flows = 160 counterfactual removals.")
    print("PASS: the metric is monotone, so responsibility is always >= 0.")


if __name__ == "__main__":
    test_no_congestion()
    test_removal_does_not_help()
    test_equal_responsibility()
    test_no_alternate_path()
    test_multiple_congested_links()
    test_monotonicity()
    print("\nAll edge-case checks passed.")
