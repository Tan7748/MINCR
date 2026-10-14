"""
responsibility.py
-----------------
Architecture Block 4: "Congestion Responsibility Calculation".

        CR_i = DeltaC_i / ( SUM_j DeltaC_j + epsilon )

CR_i is flow i's SHARE OF THE BLAME, between 0 and 1.
The epsilon (a tiny number, 1e-9) only exists to stop a division by zero
when nothing is congested and every DeltaC_j is 0.

TWO PROPERTIES WE VERIFIED (and must state honestly in the viva)
================================================================
1. CR_i >= 0 always, because our C metric is monotone (see congestion.py).
2. SUM_i DeltaC_i is generally NOT equal to C(G).
   Example: two flows each push a link from 0.5 to 0.9 utilisation
   together. Removing either one alone drops it below the threshold, so
   BOTH get a large DeltaC, and the deltas add up to more than the actual
   excess. This is the classic "overlapping cause" problem.
   => CR_i is therefore a RELATIVE blame share, not an exact decomposition
      of C(G). The exact decomposition would be the Shapley value, which
      needs 2^F coalitions and is too expensive here. We use the cheap
      one-flow-removal approximation and say so.

TIE HANDLING
============
When several flows have (nearly) equal CR, we do not pick arbitrarily.
We break ties with a documented, deterministic rule:
   (a) higher CR first,
   (b) then SMALLER demand first  -> minimum intervention: move the least
       traffic you can get away with,
   (c) then alphabetical flow id  -> fully deterministic, reproducible runs.
"""

EPSILON = 1e-9
TIE_TOLERANCE = 1e-9   # CR values closer than this count as "equal"


def responsibility_scores(deltas, flows):
    """Turn raw DeltaC values into normalised CR scores.

    deltas : dict flow_id -> {"delta": float, ...}   (from counterfactual.py)
    flows  : list of Flow objects

    Returns dict flow_id -> {"delta", "CR", "demand"}.
    """
    total = sum(info["delta"] for info in deltas.values())
    demand_of = {f.fid: f.demand for f in flows}

    scores = {}
    for fid, info in deltas.items():
        scores[fid] = {
            "delta": info["delta"],
            "CR": info["delta"] / (total + EPSILON),
            "demand": demand_of.get(fid, 0.0),
        }
    return scores


def rank_flows(scores):
    """Order flows by responsibility using the documented tie-break rule.

    Returns a list of flow ids, most responsible first.
    """
    return sorted(
        scores.keys(),
        key=lambda fid: (-scores[fid]["CR"], scores[fid]["demand"], fid),
    )


def tied_group(scores, ranked, tolerance=TIE_TOLERANCE):
    """Return all flow ids whose CR equals the top CR (within tolerance).

    Used only for reporting: we want the printout to say honestly
    "F2 and F5 are equally responsible; we chose F5 because it is smaller."
    """
    if not ranked:
        return []
    top = scores[ranked[0]]["CR"]
    return [fid for fid in ranked if abs(scores[fid]["CR"] - top) <= tolerance]


def any_responsibility(scores):
    """True if at least one flow has a non-zero DeltaC.

    False means: either there is no congestion at all, or removing any single
    flow does not help (every flow is individually irrelevant). Both are
    edge cases we must handle instead of crashing.
    """
    return any(info["delta"] > EPSILON for info in scores.values())
