# CCRR — Counterfactual Congestion Responsibility Routing (Prototype 1)

A deterministic, student-level simulation of the CCRR architecture:

```
NORMAL NETWORK → CONGESTION → COUNTERFACTUAL FLOW REMOVAL → RESPONSIBILITY SCORE
→ SELECT RESPONSIBLE FLOW → FIND ALTERNATE PATH → MINIMUM INTERVENTION
→ REROUTE → RECALCULATE → BEFORE/AFTER COMPARISON
```

No machine learning, no reinforcement learning, no external solver. Pure graph
mathematics, so every number is explainable in a viva.

---

## 1. Install and run

```bash
pip install networkx matplotlib
python main.py              # the lab demo: full pipeline + tables + figures
python experiments.py       # 5 reproducible experiments + summary table
python experiments.py -v    # same, but print every table for every experiment
python test_edge_cases.py   # proves the 6 awkward cases are handled
```

Python 3.8+. Figures are written to `./results/` as PNG (matplotlib uses the
`Agg` backend, so it works over SSH on a lab PC with no display).

## 2. What you should see

`main.py` prints eight numbered steps. On the built-in demo scenario the real
output is:

| | BEFORE | AFTER |
|---|---|---|
| C(G) | 0.2625 | **0.0000** |
| Max link utilisation | 102.5 % | 80.0 % |
| Congested links | 2 | 0 |
| Flows rerouted | 0 | 2 of 6 |

and the responsibility table:

| Flow | Demand | C(G) | C(G₋ᵢ) | ΔCᵢ | CRᵢ |
|---|---|---|---|---|---|
| F5 | 9.0 | 0.2625 | 0.0375 | 0.2250 | 0.286 |
| F4 | 12.0 | 0.2625 | 0.0375 | 0.2250 | 0.286 |
| F2 | 20.0 | 0.2625 | 0.0375 | 0.2250 | 0.286 |
| F6 | 15.0 | 0.2625 | 0.2250 | 0.0375 | 0.048 |
| F3 | 18.0 | 0.2625 | 0.2250 | 0.0375 | 0.048 |
| F1 | 34.0 | 0.2625 | 0.2250 | 0.0375 | 0.048 |

**This table is the whole project in one picture.** F1 is the *largest* flow
(34 Mbps) but has the *lowest* responsibility (CR = 0.048), because it does not
touch the bottleneck link R4–R7. A size-based heuristic would punish F1. CCRR
does not.

Baseline comparison from the same run:

| Strategy | C(G) | MLU | Flows moved | Mbps moved |
|---|---|---|---|---|
| B1: shortest path, no intervention | 0.2625 | 102.5 % | 0 | 0.0 |
| B2: reroute the **largest** flow on the worst link (F2) | 0.0375 | 83.8 % | 1 | 20.0 |
| **CCRR: reroute the most responsible flow (F5)** | **0.0000** | **80.0 %** | 2 | 24.0 |

*(All numbers above were produced by running the code. Nothing is hard-coded.)*

---

## 3. File → architecture block map

| Architecture block (your diagram) | File | Key functions |
|---|---|---|
| 1. Traffic Monitoring & Data Collection | `traffic.py` | `demo_flows`, `random_flows` |
| 2. Flow–Link Interaction Construction | `network.py`, `congestion.py` | `build_topology`, `compute_link_loads`, `compute_utilisation` |
| 3. Counterfactual Removal Operator | `counterfactual.py` | `remove_flow`, `counterfactual_states` |
| 4. Congestion Responsibility Calculation | `responsibility.py` | `responsibility_scores`, `rank_flows` |
| 5. Causal Cascade Propagation | *not implemented in Prototype 1* | future work (see §9) |
| 6. Minimum Causal Intervention Engine | `intervention.py`, `routing.py` | `minimum_intervention`, `best_reroute_for_flow`, `candidate_paths` |
| 7. Enforcement & Adaptation | `intervention.py`, `congestion.py` | applying `working_paths`, then `network_state` re-verification |
| 8. Responsibility Field Update | partial — one iteration of the loop | future work: continuous time steps |
| Outcomes / Benefits panel | `main.py`, `experiments.py` | before/after tables, baselines |
| Visualisation (for the demo) | `visualization.py` | `draw_state`, `draw_responsibility_bar` |

Be upfront with your faculty: **Blocks 5 and 8 are not in Prototype 1.** Blocks
1, 2, 3, 4, 6, 7 are.

---

## 4. The mathematics

### 4.1 Utilisation
```
u_e = load_e / capacity_e
```
A link is **congested** when `u_e > θ`, with `θ = 0.80`.

### 4.2 Congestion metric
```
C(G) = Σ_e max(0, u_e − θ)
```
Total *excess* utilisation summed over all links.

Why this one:
1. `C(G) = 0` exactly when the network is healthy — a clean baseline.
2. **Monotone.** Removing traffic can never raise a load, so `C` can never rise
   when a flow is deleted ⇒ `ΔCᵢ ≥ 0` always. Nobody can get negative blame.
   `test_edge_cases.py` case 6 verifies this over 160 removals.
3. It counts *every* congested link, so an "improvement" that shifts congestion
   from link A to link B scores as no improvement.
4. Links below θ contribute 0, so a flow on quiet links gets zero responsibility
   — which is the behaviour we want.

We also report **MLU = max_e u_e**, the standard traffic-engineering metric, so
results can be cross-checked against the literature.

### 4.3 Counterfactual removal
```
G_i⁻¹ = the same network with flow i deleted
ΔC_i  = C(G) − C(G_i⁻¹)
```

### 4.4 Responsibility score
```
CR_i = ΔC_i / ( Σ_j ΔC_j + ε ),    ε = 1e−9
```
`ε` exists only to avoid 0/0 when nothing is congested.

### 4.5 Does the formula behave correctly? — the honest answer

**Yes for sign, no for exact additivity.**

`Σᵢ ΔCᵢ ≠ C(G)` in general. In the demo run, `Σ ΔCᵢ = 0.7875` while
`C(G) = 0.2625`. This happens because several flows *jointly* cause one link's
overload: removing any one of them alone clears it, so each gets full credit and
the credits over-count. This is the classic **overlapping-cause problem**.

Consequence, and you must say this in the viva: **CRᵢ is a normalised relative
blame share, not an exact decomposition of C(G).** The exact decomposition is
the Shapley value, which needs 2^F coalitions (1024 recomputations for 10
flows) and is out of scope for Prototype 1. Our one-flow-removal delta is the
cheap first-order approximation of it.

A visible symptom of the approximation is **saturation**: in the demo, F5 (9 Mbps),
F4 (12 Mbps) and F2 (20 Mbps) all score CR = 0.286, because removing *any* of
them drops R4–R7 below the threshold. We do not hide this. We turn it into the
tie-break rule below.

### 4.6 Intervention objective
```
f(a) = ΔC(a) + α·ΔD(a) + γ·Cost(a)      α = 0.10,  γ = 0.02
ΔD(a) = (new hops − old hops) / old hops      (delay proxy)
Cost(a) = number of flows disturbed so far
```
We choose `a* = argmin f(a)` among all alternate paths, subject to the hard
constraint `ΔC(a) < 0` (strict improvement only). The fairness term `β·ΔF(a)`
from the architecture diagram is **not** implemented in Prototype 1.

### 4.7 Tie-break rule (documented and deterministic)
When CR values are equal: 1) higher CR, 2) then **smaller demand** — moving less
traffic is a smaller intervention, 3) then alphabetical flow ID for full
reproducibility. In the demo this is why CCRR moves F5 (9 Mbps) where the naive
baseline moves F2 (20 Mbps) for the same relief on that link.

---

## 5. Key functions, line by line

### `compute_link_loads(G, flows, paths)` — congestion.py
```python
loads = {link_key(u, v): 0.0 for u, v in G.edges()}   # start every link at 0
for flow in flows:
    path = paths.get(flow.fid)                        # e.g. ['R1','R2','R5','R8']
    if not path: continue                             # unroutable flow carries nothing
    for u, v in zip(path[:-1], path[1:]):             # walk the path link by link
        loads[link_key(u, v)] += flow.demand          # add this flow's Mbps
```
`zip(path[:-1], path[1:])` turns `[R1,R2,R5]` into the pairs `(R1,R2), (R2,R5)`.
`link_key` sorts the pair alphabetically so `R4–R7` and `R7–R4` are never counted
as two different links.

### `counterfactual_states(...)` — counterfactual.py
```python
factual = network_state(G, flows, paths, threshold)   # the real world
C_factual = factual["C"]
for flow in flows:
    reduced_flows = remove_flow(flows, flow.fid)      # the what-if world
    hypothetical = network_state(G, reduced_flows, paths, threshold)
    delta = C_factual - hypothetical["C"]             # ΔC_i
```
Note we keep `paths` unchanged — the other flows do **not** re-route in the
hypothetical world. That is a deliberate modelling choice: we are isolating this
one flow's contribution, not simulating a network that reconverges.

### `responsibility_scores(deltas, flows)` — responsibility.py
```python
total = sum(info["delta"] for info in deltas.values())   # Σ_j ΔC_j
scores[fid] = {"CR": info["delta"] / (total + EPSILON)}  # normalise to a share
```

### `best_reroute_for_flow(...)` — intervention.py
```python
options = candidate_paths(G, flow, current)      # k shortest loop-free alternatives
for new_path in options:
    trial_paths = dict(paths); trial_paths[flow.fid] = new_path   # pretend
    state_after = network_state(G, flows, trial_paths, threshold) # re-measure
    if state_after["C"] >= state_before["C"]: continue            # reject: no help
    cost = action_cost(...)                                       # score it
```
`dict(paths)` makes a copy, so a rejected trial never corrupts the real routing
table. This "simulate before you commit" step is Block 6's candidate evaluation.

### `minimum_intervention(...)` — intervention.py
Loops at most `max_moves` (default 2) times. Each round: find the worst
congested link → take the flows crossing it → sort them by CR → try them in
order → accept the first one with an improving reroute → stop early if `C = 0`.
Zero congestion means zero moves.

---

## 6. The five experiments

`python experiments.py`. Every scenario has a fixed seed, so your report and the
faculty's re-run will match exactly.

| Exp | Flows | Demand | C before | C naive (B2) | C CCRR | MLU before | MLU after | Moved | Status |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 6 | 108.0 | 0.2625 | 0.0375 | **0.0000** | 102.5 % | 80.0 % | 2 | resolved |
| E2 | 5 | 65.7 | 0.0000 | 0.0000 | 0.0000 | 54.2 % | 54.2 % | 0 | healthy |
| E3 | 7 | 179.4 | 0.0863 | 0.0000 | 0.0000 | 88.6 % | 73.5 % | 1 | resolved |
| E4 | 9 | 257.9 | 0.3904 | 0.0550 | 0.0550 | 109.2 % | 85.5 % | 1 | partially_resolved |
| E5 | 10 | 380.9 | 2.6682 | 1.7337 | **1.3125** | 184.5 % | 146.5 % | 2 | partially_resolved |

CCRR reduced C(G) in 4/5 experiments (E2 needed no action) and beat the naive
largest-flow baseline in 2/5. **Report this honestly.** E3 and E4 are ties with
the baseline — in those scenarios the most responsible flow happens to be the
largest one. E5 is a genuine overload where no single-path rerouting can fix
everything; CCRR relieves it partially and says so. A result table that showed
CCRR winning 5/5 would be the suspicious one.

Edge cases covered by `test_edge_cases.py`: no congestion, single removal is
insufficient, exactly equal responsibility, no alternate path, multiple
congested links, and monotonicity of ΔC over 160 random removals.

---

## 7. Why this structure, and one simplification we made

We kept the 10 modules you specified — the one-file-per-block layout maps
directly onto the architecture diagram, which is worth more in a viva than
saving a few imports. Two deliberate deviations:

- We **added** `test_edge_cases.py`, because "we handled the edge cases" is much
  stronger when it is executable.
- We **did not** create a separate module for Block 5 (causal cascade), because
  writing an empty file would be pretending we implemented it.

Modelling simplifications, all stated on purpose:
- Links are **undirected**; one link carries the sum of both directions.
- Flows are **unsplittable** — one flow takes one path.
- Traffic is a **static rate**, not packets. There is no queueing, no TCP
  reaction, no propagation delay.
- One control iteration, not a continuous feedback loop.

---

## 8. 80–90 % completion checklist

| # | Requirement | Status |
|---|---|---|
| 1 | Python implementation | done |
| 2 | NetworkX + Matplotlib | done |
| 3 | 8-node topology | done |
| 4 | Link capacities | done |
| 5 | ≥ 5 flows with src/dst/demand | done (6 demo, up to 10 in E5) |
| 6 | Routing | done (shortest path + k-shortest alternates) |
| 7 | Link loads and utilisation | done |
| 8 | Congestion threshold | done (θ = 0.80, one constant) |
| 9 | Congestion metric C(G) | done, with justification |
| 10 | Counterfactual removal per flow | done |
| 11 | Normalised responsibility scores | done, with honest caveats |
| 12 | Identify responsible flows on the congested link | done |
| 13 | Alternate feasible path | done |
| 14 | Minimum-intervention rerouting | done (fewest flows, smallest flow on ties) |
| 15 | Recalculate after rerouting | done |
| 16 | Before/after comparison | done |
| 17 | All required tables | done |
| 18 | Before/after visualisations | done |
| 19 | Baseline | done (two: shortest-path, and largest-flow heuristic) |
| 20 | 5 reproducible experiments | done (fixed seeds) |
| 21 | Real generated results | done (nothing hard-coded) |
| 22 | Edge cases | done + tested |
| — | Block 5: causal cascade propagation | **not done** |
| — | Block 8: continuous responsibility-field update | **not done** |
| — | Fairness term β·ΔF in the objective | **not done** |

That is roughly **85 %** of the stated scope: the counterfactual–responsibility–
minimum-intervention core is complete end to end; the temporal/causal-cascade
layer is not.

---

## 9. Honest future work

1. **Block 5 — causal cascade.** Multi-time-step simulation where relieving link
   A pushes load onto link B, with a propagation-gain metric. Needs a time loop,
   which Prototype 1 does not have.
2. **Block 8 — closed loop.** Re-measure, update CR, intervene again, repeat;
   check for oscillation (the classic danger of reactive rerouting).
3. **Shapley-style exact attribution** for small flow sets, to quantify how far
   our one-removal approximation is from the exact decomposition.
4. **Fairness term** β·ΔF, e.g. Jain's fairness index over per-flow delay.
5. **Directed links and queueing**, so delay is modelled rather than proxied by
   hop count.
6. **Real traffic**: replay a public trace, or run it over Mininet/Ryu so the
   controller pushes real flow rules.
7. **Scalability**: our counterfactual loop is O(F) full recomputations. An
   incremental load update would cut this substantially for large F.

Claim 1–7 as future work, not as things that "almost work".

---

## 10. Development roadmap

**Prototype 1 (this code) — done.** Static simulation, counterfactual attribution,
single-round minimum intervention, baselines, 5 experiments.

**Prototype 2 — next.**
- Add the time dimension: T rounds of measure → attribute → intervene.
- Implement Block 5 (cascade) and Block 8 (responsibility-field update).
- Add the fairness term to `f(a)`.
- Add a third baseline: full re-optimisation (reroute everything), to show how
  much disruption CCRR saves for how much congestion relief.
- Scale to 20–30 nodes and 50+ flows; plot runtime vs number of flows.

**Final project.**
- Port to Mininet + a Ryu/POX SDN controller so interventions become real
  OpenFlow rules.
- Evaluate on a standard topology (e.g. Abilene) and a realistic traffic matrix.
- Report: congestion relief vs flows disturbed, compared against shortest-path,
  largest-flow, and full re-optimisation.
- Write up the approximation gap versus Shapley attribution.

---

## 11. Viva script

**"What is your project?"**
> Normal routing reacts to a congested link by moving traffic, but it never asks
> *which* flow is actually causing the congestion. We compute that. For every
> flow we run a counterfactual test — recompute the network as if that flow did
> not exist — and measure how much congestion disappears. That gives each flow a
> responsibility score. We then apply the smallest intervention we can: reroute
> only the most responsible flow, and leave everything else untouched.

**"What is counterfactual analysis?"**
> Asking a what-if question instead of just reading the current state.
> "What would congestion be if flow F5 weren't there?" We answer it by literally
> deleting F5 from the simulation and recomputing every link load. The
> difference between the real congestion and that hypothetical congestion is
> ΔC, the flow's measured contribution.

**"How is responsibility calculated?"**
> C(G) is total excess utilisation above 80 %. ΔCᵢ = C(G) − C(G without flow i).
> CRᵢ = ΔCᵢ divided by the sum of all ΔC values, so the scores form a
> normalised share. Because our metric is monotone, ΔCᵢ can never be negative.

**"How is this different from shortest-path routing?"**
> Shortest path is completely blind to load — it computed the congestion in the
> first place. In our demo it drove link R4–R7 to 102.5 %. We start from that
> state and repair it with two reroutes, reaching zero congested links.

**"How is this different from ordinary congestion-aware routing?"**
> Ordinary congestion-aware routing is *symptom-driven*: it sees a full link and
> moves whatever is convenient, usually the biggest flow. Ours is
> *attribution-driven*. In our demo the biggest flow, F1 at 34 Mbps, has the
> lowest responsibility, 0.048, because it never touches the bottleneck. A
> size-based policy would move 34 Mbps for no benefit. We moved a 9 Mbps flow
> instead and cleared the link, because the counterfactual test showed it was
> equally responsible at a third of the cost.

**"What exactly have you implemented?"**
> Blocks 1, 2, 3, 4, 6 and 7 of the architecture: monitoring/traffic generation,
> flow–link mapping, the counterfactual removal operator, the responsibility
> calculation, the minimum-intervention engine with its cost function, and
> enforcement with post-intervention verification. Two baselines, five seeded
> experiments, and six edge-case tests. Blocks 5 and 8 — causal cascade
> propagation and the continuous responsibility-field update — are future work.

**"What is your novelty claim?"**
> We are not claiming the concept is unprecedented. Counterfactual what-if
> analysis, contribution-based responsibility scoring, and minimum-intervention
> traffic engineering each exist separately — in MPLS traffic engineering,
> power-grid congestion cost allocation, and causal root-cause analysis for
> networks. Our contribution is the integration: a single closed pipeline where
> a per-flow counterfactual removal test produces an explicit responsibility
> metric that directly drives a minimum-intervention routing decision, plus a
> working, reproducible open implementation for packet networks.

**"What are your limitations?"**
> Four honest ones. First, this is counterfactual attribution *inside a
> simulation* — valid because we can actually re-run the network without a flow,
> which you cannot do in production. It is not a claim of real-world causal
> certainty. Second, ΔC values do not sum to C(G) because causes overlap, so CR
> is a relative blame share, not an exact decomposition; the exact version is the
> Shapley value, which is exponential. Third, the model is static: no packets, no
> queueing, no TCP dynamics, undirected links, unsplittable flows. Fourth, the
> counterfactual loop costs one full recomputation per flow, which will not scale
> to thousands of flows without an incremental method.
