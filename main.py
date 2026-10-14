"""
main.py
-------
Runs the complete CCRR pipeline once, on the fixed demo scenario, and prints
every table the faculty asked for.

PIPELINE (this is the whole project in 9 steps)
    NORMAL NETWORK
 -> CONGESTION
 -> COUNTERFACTUAL FLOW REMOVAL
 -> RESPONSIBILITY SCORE
 -> SELECT RESPONSIBLE FLOW
 -> FIND ALTERNATE PATH
 -> MINIMUM INTERVENTION
 -> REROUTE + RECALCULATE
 -> BEFORE / AFTER COMPARISON

Run it with:   python main.py
"""

import os

from network import build_topology, describe_topology, link_key
from traffic import demo_flows, total_demand
from congestion import network_state, CONGESTION_THRESHOLD, flows_on_link
from counterfactual import counterfactual_states
from responsibility import responsibility_scores, rank_flows, tied_group, any_responsibility
from routing import shortest_path_routing, path_str
from intervention import minimum_intervention, largest_flow_baseline, describe_move
from visualization import draw_state, draw_responsibility_bar

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Printing helpers (plain text tables, no extra libraries needed)
# ---------------------------------------------------------------------------
def header(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def flow_table(flows, paths, title="FLOW TABLE"):
    print(f"\n{title}")
    print(f"{'Flow':<6}{'Src':<5}{'Dst':<5}{'Demand':>9}   Path")
    print("-" * 78)
    for f in flows:
        print(f"{f.fid:<6}{f.src:<5}{f.dst:<5}{f.demand:>8.1f}   {path_str(paths.get(f.fid))}")


def link_table(G, state, title="LINK TABLE"):
    print(f"\n{title}")
    print(f"{'Link':<12}{'Capacity':>10}{'Load':>9}{'Util%':>9}   Status")
    print("-" * 78)
    for link in sorted(state["util"], key=lambda k: state["util"][k], reverse=True):
        u, v = link
        util = state["util"][link]
        status = "CONGESTED" if util > state["threshold"] else ""
        print(f"{u + '--' + v:<12}{G[u][v]['capacity']:>10.1f}"
              f"{state['loads'][link]:>9.1f}{util * 100:>8.1f}%   {status}")
    print("-" * 78)
    print(f"C(G) = {state['C']:.4f}    MLU = {state['MLU'] * 100:.1f}%    "
          f"congested links = {len(state['congested'])}")


def responsibility_table(scores, ranked, deltas, C_factual):
    print("\nCOUNTERFACTUAL + RESPONSIBILITY TABLE")
    print(f"{'Flow':<6}{'Demand':>8}{'C(G)':>10}{'C(G_-i)':>11}"
          f"{'DeltaC_i':>11}{'CR_i':>9}")
    print("-" * 78)
    for fid in ranked:
        print(f"{fid:<6}{scores[fid]['demand']:>8.1f}{C_factual:>10.4f}"
              f"{deltas[fid]['C_without']:>11.4f}"
              f"{scores[fid]['delta']:>11.4f}{scores[fid]['CR']:>9.3f}")
    print("-" * 78)
    print(f"sum of DeltaC_i = {sum(s['delta'] for s in scores.values()):.4f}   "
          f"(note: this need NOT equal C(G); CR is a relative blame share)")


# ---------------------------------------------------------------------------
# The reusable pipeline. experiments.py calls this too.
# ---------------------------------------------------------------------------
def run_pipeline(G, flows, threshold=CONGESTION_THRESHOLD, verbose=True,
                 max_moves=2):
    """Execute the full CCRR pipeline and return a result dictionary."""

    # --- STEP 1: baseline shortest-path routing (Block 2) -------------------
    paths_before = shortest_path_routing(G, flows)
    state_before = network_state(G, flows, paths_before, threshold)

    if verbose:
        flow_table(flows, paths_before, "STEP 1 - FLOWS ROUTED BY SHORTEST PATH (BASELINE)")
        link_table(G, state_before, "STEP 2 - LINK LOADS AND CONGESTION (BEFORE)")

    # --- EDGE CASE: nothing is congested -----------------------------------
    if not state_before["congested"]:
        if verbose:
            print("\nNo link exceeds the threshold. C(G) = 0, so the correct "
                  "minimum intervention is to DO NOTHING.")
        return {
            "flows": flows, "paths_before": paths_before,
            "paths_after": paths_before, "state_before": state_before,
            "state_after": state_before, "scores": {}, "ranked": [],
            "log": [], "status": "healthy", "baseline_info": None,
            "baseline_state": state_before,
            "disturbed_mbps": 0.0, "baseline_mbps": 0.0,
        }

    # --- STEP 3: counterfactual removal (Block 3) --------------------------
    C_factual, deltas = counterfactual_states(G, flows, paths_before, threshold)

    # --- STEP 4: responsibility scores (Block 4) ---------------------------
    scores = responsibility_scores(deltas, flows)
    ranked = rank_flows(scores)

    if verbose:
        responsibility_table(scores, ranked, deltas, C_factual)

    if not any_responsibility(scores):
        if verbose:
            print("\nEDGE CASE: removing any SINGLE flow does not reduce C(G). "
                  "The congestion is caused by overlapping flows, none of which "
                  "is individually sufficient. Single-flow attribution is "
                  "undefined here; we fall back to moving any flow on the worst "
                  "link that produces an improvement.")

    # --- STEP 5: which flows sit on the worst link? ------------------------
    worst_link = state_before["congested"][0][0]
    on_worst = [f.fid for f in flows_on_link(flows, paths_before, worst_link)]
    tied = tied_group(scores, [f for f in ranked if f in on_worst])

    if verbose:
        print(f"\nSTEP 3 - WORST CONGESTED LINK: {worst_link[0]}--{worst_link[1]} "
              f"at {state_before['congested'][0][1] * 100:.1f}% utilisation")
        print(f"          flows crossing it: {', '.join(on_worst)}")
        if len(tied) > 1:
            chosen = min(tied, key=lambda f: (scores[f]["demand"], f))
            print(f"          TIE: {', '.join(tied)} all have CR = "
                  f"{scores[tied[0]]['CR']:.3f}. Documented tie-break = move the "
                  f"SMALLEST flow -> {chosen} ({scores[chosen]['demand']:.1f} Mbps)")

    # --- STEPS 6-8: minimum intervention (Blocks 6 + 7) --------------------
    paths_after, log, status = minimum_intervention(
        G, flows, paths_before, threshold, scores, ranked, max_moves=max_moves)
    state_after = network_state(G, flows, paths_after, threshold)

    # --- BASELINE 2: naive "move the biggest flow" -------------------------
    baseline_paths, baseline_info = largest_flow_baseline(
        G, flows, paths_before, threshold)
    baseline_state = network_state(G, flows, baseline_paths, threshold)

    # How much traffic did each strategy actually disturb? This is the most
    # direct measure of "minimum intervention".
    demand_of = {f.fid: f.demand for f in flows}
    disturbed = sum(demand_of[e["flow"]] for e in log)
    baseline_mbps = (demand_of[baseline_info["flow"]]
                     if baseline_info and baseline_info.get("moved") else 0.0)

    if verbose:
        print("\nSTEP 4 - MINIMUM INTERVENTION DECISION")
        if not log:
            print("   No feasible improving reroute was found "
                  f"(status = {status}). The network is left unchanged.")
        for entry in log:
            print("   " + describe_move(entry))
        print(f"   flows in network = {len(flows)}   flows moved = {len(log)}"
              f"   ({len(log) / max(1, len(flows)) * 100:.0f}% of traffic disturbed)")

        flow_table(flows, paths_after, "STEP 5 - FLOW TABLE (AFTER INTERVENTION)")
        link_table(G, state_after, "STEP 6 - LINK LOADS AND CONGESTION (AFTER)")

        print("\nSTEP 7 - BEFORE vs AFTER")
        print(f"{'Metric':<28}{'BEFORE':>12}{'AFTER':>12}{'CHANGE':>14}")
        print("-" * 78)
        dC = state_after['C'] - state_before['C']
        dM = state_after['MLU'] - state_before['MLU']
        print(f"{'Congestion C(G)':<28}{state_before['C']:>12.4f}"
              f"{state_after['C']:>12.4f}{dC:>+14.4f}")
        print(f"{'Max link utilisation':<28}{state_before['MLU'] * 100:>11.1f}%"
              f"{state_after['MLU'] * 100:>11.1f}%{dM * 100:>+13.1f}%")
        print(f"{'Congested links':<28}{len(state_before['congested']):>12}"
              f"{len(state_after['congested']):>12}"
              f"{len(state_after['congested']) - len(state_before['congested']):>+14}")
        print(f"{'Flows rerouted':<28}{0:>12}{len(log):>12}")
        print(f"{'Traffic disturbed (Mbps)':<28}{0.0:>12.1f}{disturbed:>12.1f}")

        print("\nSTEP 8 - COMPARISON AGAINST BASELINES")
        print(f"{'Strategy':<42}{'C(G)':>9}{'MLU':>9}{'moved':>7}{'Mbps moved':>12}")
        print("-" * 79)
        print(f"{'B1: shortest path, no intervention':<42}"
              f"{state_before['C']:>9.4f}{state_before['MLU'] * 100:>8.1f}%"
              f"{0:>7}{0.0:>12.1f}")
        moved_b2 = 1 if (baseline_info and baseline_info.get("moved")) else 0
        b2_name = (f"B2: reroute LARGEST flow ({baseline_info['flow']})"
                   if baseline_info else "B2: reroute LARGEST flow")
        print(f"{b2_name:<42}"
              f"{baseline_state['C']:>9.4f}{baseline_state['MLU'] * 100:>8.1f}%"
              f"{moved_b2:>7}{baseline_mbps:>12.1f}")
        ours = (f"CCRR: reroute MOST RESPONSIBLE ({log[0]['flow']})"
                if log else "CCRR: no action possible")
        print(f"{ours:<42}"
              f"{state_after['C']:>9.4f}{state_after['MLU'] * 100:>8.1f}%"
              f"{len(log):>7}{disturbed:>12.1f}")

    return {
        "flows": flows,
        "paths_before": paths_before,
        "paths_after": paths_after,
        "state_before": state_before,
        "state_after": state_after,
        "scores": scores,
        "ranked": ranked,
        "log": log,
        "status": status,
        "baseline_info": baseline_info,
        "baseline_state": baseline_state,
        "disturbed_mbps": disturbed,
        "baseline_mbps": baseline_mbps,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    header("COUNTERFACTUAL CONGESTION RESPONSIBILITY ROUTING (CCRR) - PROTOTYPE 1")
    G = build_topology()
    flows = demo_flows()

    print(describe_topology(G))
    print(f"\nTraffic: {len(flows)} flows, total demand "
          f"{total_demand(flows):.1f} Mbps")
    print(f"Congestion threshold: a link is congested above "
          f"{CONGESTION_THRESHOLD * 100:.0f}% utilisation")

    result = run_pipeline(G, flows)

    # ---- figures ----------------------------------------------------------
    moved = result["log"][0]["flow"] if result["log"] else None
    f1 = draw_state(G, flows, result["paths_before"], result["state_before"],
                    "BEFORE: shortest-path routing",
                    os.path.join(RESULTS_DIR, "demo_before.png"))
    f2 = draw_state(G, flows, result["paths_after"], result["state_after"],
                    "AFTER: CCRR minimum intervention",
                    os.path.join(RESULTS_DIR, "demo_after.png"),
                    highlight_flow=moved)
    figs = [f1, f2]
    if result["ranked"]:
        figs.append(draw_responsibility_bar(
            result["scores"], result["ranked"],
            os.path.join(RESULTS_DIR, "demo_responsibility.png"),
            "Congestion responsibility per flow (counterfactual removal)"))

    print("\nFigures written:")
    for f in figs:
        print("   " + f)
    print("\nDone. Run 'python experiments.py' for the 5 reproducible experiments.")


if __name__ == "__main__":
    main()
