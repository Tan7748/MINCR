"""
intervention.py
---------------
Architecture Blocks 6 + 7: "Minimum Causal Intervention Engine" and
"Enforcement & Adaptation".

WHAT "MINIMUM INTERVENTION" MEANS HERE
======================================
We do NOT re-optimise the whole network. We try to fix the problem by
moving ONE flow. Only if a single move cannot help do we allow a second
move, and so on, up to `max_moves`. The search stops the moment the
network is healthy (C = 0).

For each candidate flow (taken in responsibility order) we evaluate every
alternate path and score it with a simplified version of the cost function
in the architecture diagram:

        f(a) = DeltaC(a) + alpha * DeltaD(a) + gamma * Cost(a)

where
        DeltaC(a) = C_after - C_before      (congestion change; negative = good)
        DeltaD(a) = extra hops / original hops  (delay proxy)
        Cost(a)   = 1 per rerouted flow      (operational disruption)

We pick the action with the LOWEST f(a), and we only accept it if it
actually reduces congestion (DeltaC < 0). Fairness (beta * DeltaF) is left
out of Prototype 1 on purpose - see "future work" in the README.
"""

from congestion import network_state, flows_on_link
from routing import candidate_paths, path_str

ALPHA = 0.10   # weight on the delay/stretch penalty
GAMMA = 0.02   # weight on "number of flows we disturbed"


def action_cost(C_before, C_after, old_path, new_path, n_moves):
    """The objective f(a). Lower is better."""
    delta_C = C_after - C_before
    old_hops = max(1, len(old_path) - 1)
    new_hops = max(1, len(new_path) - 1)
    delta_D = (new_hops - old_hops) / old_hops
    return delta_C + ALPHA * delta_D + GAMMA * n_moves


def best_reroute_for_flow(G, flows, paths, threshold, flow, n_moves=1):
    """Try every alternate path for ONE flow; return the best improving one.

    Returns (new_path, new_state, cost) or (None, None, None) when this flow
    has no alternate path, or no alternate path that helps.
    """
    state_before = network_state(G, flows, paths, threshold)
    current = paths.get(flow.fid)
    if current is None:
        return None, None, None

    options = candidate_paths(G, flow, current)
    if not options:
        return None, None, None      # EDGE CASE: no alternate path exists

    best = (None, None, None)
    for new_path in options:
        trial_paths = dict(paths)
        trial_paths[flow.fid] = new_path
        state_after = network_state(G, flows, trial_paths, threshold)

        # Hard constraint: never accept a move that makes congestion worse
        # or leaves it unchanged. Strict improvement only.
        if state_after["C"] >= state_before["C"] - 1e-12:
            continue

        cost = action_cost(state_before["C"], state_after["C"],
                           current, new_path, n_moves)
        if best[2] is None or cost < best[2]:
            best = (new_path, state_after, cost)

    return best


def minimum_intervention(G, flows, paths, threshold, scores, ranked,
                         max_moves=2):
    """Blocks 6 + 7: choose and apply the smallest helpful set of reroutes.

    Returns (new_paths, log) where `log` is a list of dictionaries describing
    each accepted move, plus a "status" string explaining what happened.
    """
    flow_by_id = {f.fid: f for f in flows}
    working_paths = dict(paths)
    log = []
    status = "no_action"

    for move_number in range(1, max_moves + 1):
        state = network_state(G, flows, working_paths, threshold)

        # EDGE CASE: nothing is congested -> do nothing. Minimum intervention
        # means zero intervention when the network is already healthy.
        if not state["congested"]:
            status = "healthy" if move_number == 1 else "resolved"
            break

        # Focus on the WORST congested link, but rank candidates by their
        # global responsibility score (so we never fix link A by wrecking B).
        worst_link = state["congested"][0][0]
        on_link = {f.fid for f in flows_on_link(flows, working_paths, worst_link)}

        candidates = [fid for fid in ranked
                      if fid in on_link and scores[fid]["delta"] > 0]
        if not candidates:
            # EDGE CASE: no single flow removal reduces congestion
            # (overlapping causes). Fall back to any flow on the link.
            candidates = [fid for fid in ranked if fid in on_link]

        applied = False
        for fid in candidates:
            new_path, new_state, cost = best_reroute_for_flow(
                G, flows, working_paths, threshold, flow_by_id[fid], move_number)
            if new_path is None:
                continue     # this flow cannot help; try the next one

            before_state = network_state(G, flows, working_paths, threshold)
            log.append({
                "move": move_number,
                "flow": fid,
                "target_link": worst_link,
                "CR": scores[fid]["CR"],
                "old_path": working_paths[fid],
                "new_path": new_path,
                "C_before": before_state["C"],
                "C_after": new_state["C"],
                "MLU_before": before_state["MLU"],
                "MLU_after": new_state["MLU"],
                "cost": cost,
            })
            working_paths[fid] = new_path
            applied = True
            status = "intervened"
            break

        if not applied:
            # EDGE CASE: congestion exists but nothing we can move helps.
            status = "no_feasible_action"
            break

        if network_state(G, flows, working_paths, threshold)["C"] <= 1e-12:
            status = "resolved"
            break

    # Final, unambiguous status for the report:
    #   healthy            - nothing was congested to begin with
    #   resolved           - congestion fully removed
    #   partially_resolved - some moves helped, congestion remains
    #   no_feasible_action - congested, but no reroute improves anything
    final_C = network_state(G, flows, working_paths, threshold)["C"]
    if status != "healthy":
        if final_C <= 1e-12:
            status = "resolved"
        elif log:
            status = "partially_resolved"
        else:
            status = "no_feasible_action"

    return working_paths, log, status


# ---------------------------------------------------------------------------
# BASELINE 2: reroute the LARGEST flow instead of the most RESPONSIBLE one.
# This is the comparison that shows why responsibility scoring matters.
# ---------------------------------------------------------------------------
def largest_flow_baseline(G, flows, paths, threshold):
    """Naive heuristic: on the worst link, move the biggest flow."""
    state = network_state(G, flows, paths, threshold)
    if not state["congested"]:
        return dict(paths), None

    worst_link = state["congested"][0][0]
    on_link = flows_on_link(flows, paths, worst_link)
    if not on_link:
        return dict(paths), None

    biggest = max(on_link, key=lambda f: (f.demand, f.fid))
    new_path, new_state, cost = best_reroute_for_flow(
        G, flows, paths, threshold, biggest, 1)

    if new_path is None:
        return dict(paths), {"flow": biggest.fid, "moved": False}

    out = dict(paths)
    out[biggest.fid] = new_path
    return out, {
        "flow": biggest.fid,
        "moved": True,
        "old_path": paths[biggest.fid],
        "new_path": new_path,
        "C_after": new_state["C"],
        "MLU_after": new_state["MLU"],
    }


def describe_move(entry):
    """One-line human summary of an accepted intervention."""
    return (f"move #{entry['move']}: reroute {entry['flow']} "
            f"(CR={entry['CR']:.3f})  "
            f"{path_str(entry['old_path'])}  ==>  {path_str(entry['new_path'])}")
