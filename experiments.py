"""
experiments.py
--------------
Five reproducible experiments under different traffic conditions.

Every experiment has a FIXED SEED, so the numbers in your report will be
identical every time you or your faculty run the code. Nothing is
hard-coded: every number printed here is computed by the program.

Run with:   python experiments.py
"""

import os

from network import build_topology
from traffic import demo_flows, random_flows, total_demand
from congestion import CONGESTION_THRESHOLD
from main import run_pipeline
from visualization import draw_state

RESULTS_DIR = "results"


EXPERIMENTS = [
    # (name, description, builder)
    ("E1", "Fixed demo traffic, 6 flows (the lab demo scenario)",
     lambda G: demo_flows()),

    ("E2", "Light random load, 5 flows, demand 5-20 Mbps (seed=1)",
     lambda G: random_flows(G, 5, seed=1, min_demand=5, max_demand=20)),

    ("E3", "Medium random load, 7 flows, demand 10-35 Mbps (seed=2)",
     lambda G: random_flows(G, 7, seed=2, min_demand=10, max_demand=35)),

    ("E4", "Heavy random load, 9 flows, demand 15-45 Mbps (seed=3)",
     lambda G: random_flows(G, 9, seed=3, min_demand=15, max_demand=45)),

    ("E5", "Overload, 10 flows, demand 25-55 Mbps (seed=4)",
     lambda G: random_flows(G, 10, seed=4, min_demand=25, max_demand=55)),
]


def run_all(verbose_each=False, save_figures=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    G = build_topology()
    rows = []

    for name, description, builder in EXPERIMENTS:
        flows = builder(G)
        print("\n" + "#" * 78)
        print(f"# {name}: {description}")
        print(f"# {len(flows)} flows, total demand {total_demand(flows):.1f} Mbps")
        print("#" * 78)
        for f in flows:
            print(f"   {f}")

        result = run_pipeline(G, flows, CONGESTION_THRESHOLD,
                              verbose=verbose_each)

        before, after = result["state_before"], result["state_after"]
        base = result["baseline_state"]
        moved = result["log"][0]["flow"] if result["log"] else "-"

        print(f"\n   status                : {result['status']}")
        print(f"   C(G) before           : {before['C']:.4f}   "
              f"MLU {before['MLU'] * 100:.1f}%   "
              f"congested links {len(before['congested'])}")
        print(f"   C(G) after  (CCRR)    : {after['C']:.4f}   "
              f"MLU {after['MLU'] * 100:.1f}%   "
              f"congested links {len(after['congested'])}   "
              f"flows moved {len(result['log'])}")
        print(f"   C(G) baseline B2      : {base['C']:.4f}   "
              f"MLU {base['MLU'] * 100:.1f}%   "
              f"(naive: move largest flow)")
        if result["ranked"]:
            top = result["ranked"][0]
            print(f"   most responsible flow : {top} "
                  f"(CR = {result['scores'][top]['CR']:.3f}, "
                  f"demand = {result['scores'][top]['demand']:.1f} Mbps)")
        print(f"   flow rerouted by CCRR : {moved}")

        if save_figures and before["congested"]:
            draw_state(G, flows, result["paths_before"], before,
                       f"{name} BEFORE (shortest path)",
                       os.path.join(RESULTS_DIR, f"{name}_before.png"))
            draw_state(G, flows, result["paths_after"], after,
                       f"{name} AFTER (CCRR minimum intervention)",
                       os.path.join(RESULTS_DIR, f"{name}_after.png"),
                       highlight_flow=None if moved == "-" else moved)

        rows.append({
            "name": name,
            "flows": len(flows),
            "demand": total_demand(flows),
            "C_before": before["C"],
            "C_ccrr": after["C"],
            "C_base": base["C"],
            "MLU_before": before["MLU"],
            "MLU_ccrr": after["MLU"],
            "MLU_base": base["MLU"],
            "moved": len(result["log"]),
            "flow": moved,
            "status": result["status"],
        })

    summary(rows)
    return rows


def summary(rows):
    print("\n" + "=" * 100)
    print("SUMMARY OF ALL EXPERIMENTS   (all numbers produced by this program)")
    print("=" * 100)
    print(f"{'Exp':<5}{'Flows':>6}{'Demand':>9}{'C before':>10}"
          f"{'C naive':>10}{'C CCRR':>10}{'MLU bef':>9}{'MLU aft':>9}"
          f"{'Moved':>7}{'Flow':>7}  Status")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<5}{r['flows']:>6}{r['demand']:>9.1f}"
              f"{r['C_before']:>10.4f}{r['C_base']:>10.4f}{r['C_ccrr']:>10.4f}"
              f"{r['MLU_before'] * 100:>8.1f}%{r['MLU_ccrr'] * 100:>8.1f}%"
              f"{r['moved']:>7}{r['flow']:>7}  {r['status']}")
    print("-" * 100)
    improved = [r for r in rows if r["C_ccrr"] < r["C_before"] - 1e-12]
    print(f"Experiments where CCRR reduced C(G): {len(improved)} / {len(rows)}")
    beat = [r for r in rows if r["C_ccrr"] < r["C_base"] - 1e-12]
    print(f"Experiments where CCRR beat the naive largest-flow baseline: "
          f"{len(beat)} / {len(rows)}")
    print("Figures saved in ./results/")


if __name__ == "__main__":
    import sys
    run_all(verbose_each="-v" in sys.argv)
