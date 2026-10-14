"""
traffic.py
----------
Architecture Block 1: "Traffic Monitoring & Data Collection".

In a real network this module would read counters from routers.
In our simulation we *define* the traffic ourselves, which is honest and
reproducible: we know the ground truth, so we can check whether the
responsibility scores make sense.

A Flow = (source router, destination router, demand in Mbps).
Example: F1 sends 45 Mbps from R1 to R8.
"""

import random
from dataclasses import dataclass


@dataclass
class Flow:
    """One traffic flow."""
    fid: str        # flow id, e.g. "F1"
    src: str        # source node
    dst: str        # destination node
    demand: float   # Mbps

    def __repr__(self):
        return f"{self.fid}({self.src}->{self.dst}, {self.demand:.1f} Mbps)"


# ---------------------------------------------------------------------------
# Scenario 0: the hand-made demo scenario used by main.py.
# It is hand-made (not random) so the lab demo always shows the same,
# easy-to-explain story.
# ---------------------------------------------------------------------------
DEMO_FLOWS = [
    Flow("F1", "R1", "R8", 34.0),   # biggest flow, but NOT the most responsible
    Flow("F2", "R1", "R7", 20.0),
    Flow("F3", "R2", "R8", 18.0),
    Flow("F4", "R3", "R7", 12.0),
    Flow("F5", "R2", "R7", 9.0),    # smallest flow, yet equally responsible
    Flow("F6", "R1", "R5", 15.0),
]


def demo_flows():
    """Return a fresh copy of the fixed demo traffic."""
    return [Flow(f.fid, f.src, f.dst, f.demand) for f in DEMO_FLOWS]


def random_flows(G, n_flows, seed, min_demand=5.0, max_demand=40.0):
    """Generate n_flows random source->destination flows.

    `seed` fixes the random number generator, so running the same experiment
    twice gives EXACTLY the same traffic. This is what makes our experiments
    reproducible, which the faculty asked for.
    """
    rng = random.Random(seed)
    nodes = sorted(G.nodes())
    flows = []
    for i in range(n_flows):
        src, dst = rng.sample(nodes, 2)          # sample => src != dst
        demand = round(rng.uniform(min_demand, max_demand), 1)
        flows.append(Flow(f"F{i + 1}", src, dst, demand))
    return flows


def total_demand(flows):
    """Sum of all flow demands (Mbps) - used in the experiment summary."""
    return sum(f.demand for f in flows)
