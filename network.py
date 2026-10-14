"""
network.py
----------
Architecture Block 2 (part 1): builds the network graph G(V, E).

A "topology" here is just a set of routers (nodes) and links (edges).
Every link has a capacity in Mbps. Capacity = how much traffic the link
can carry before it becomes congested.

We use an undirected graph for simplicity: a link R1--R2 carries the sum
of traffic in both directions. This is a standard simplification for a
first prototype and is stated as a limitation in the README.
"""

import networkx as nx


# Each tuple is (node_a, node_b, capacity_in_Mbps)
DEFAULT_LINKS = [
    ("R1", "R2", 100),
    ("R1", "R3", 100),
    ("R2", "R4", 60),
    ("R2", "R5", 80),
    ("R3", "R4", 100),
    ("R3", "R6", 80),
    ("R4", "R5", 60),
    ("R4", "R7", 40),   # deliberately thin link -> natural bottleneck
    ("R5", "R7", 60),
    ("R5", "R8", 80),
    ("R6", "R7", 100),
    ("R6", "R8", 50),
    ("R7", "R8", 100),
]

# Fixed coordinates so that the "before" and "after" pictures look identical
# except for the routes. Without this, matplotlib would shuffle the nodes and
# the two figures would be impossible to compare.
NODE_POSITIONS = {
    "R1": (0.0, 1.0),
    "R2": (1.0, 1.8),
    "R3": (1.0, 0.2),
    "R4": (2.0, 1.0),
    "R5": (3.0, 1.8),
    "R6": (3.0, 0.0),
    "R7": (4.0, 1.0),
    "R8": (5.0, 0.9),
}


def build_topology(links=None):
    """Create the network graph.

    Returns a networkx.Graph where every edge has:
        capacity : float  (Mbps the link can carry)
        weight   : float  (routing cost; 1 = plain hop count)
    """
    if links is None:
        links = DEFAULT_LINKS

    G = nx.Graph()
    for a, b, capacity in links:
        G.add_edge(a, b, capacity=float(capacity), weight=1.0)

    # Attach the drawing coordinates to the graph so visualization.py can use them.
    for node in G.nodes():
        G.nodes[node]["pos"] = NODE_POSITIONS.get(node, (0.0, 0.0))
    return G


def link_key(u, v):
    """Give every undirected link ONE canonical name.

    ("R4", "R7") and ("R7", "R4") are the same physical link, so we always
    sort them alphabetically. Without this, the same link could be counted
    twice in the load table.
    """
    return tuple(sorted((u, v)))


def capacity_of(G, u, v):
    """Capacity of the link between u and v, in Mbps."""
    return G[u][v]["capacity"]


def describe_topology(G):
    """Human-readable summary, printed at the start of every run."""
    lines = [
        f"Nodes ({G.number_of_nodes()}): {', '.join(sorted(G.nodes()))}",
        f"Links ({G.number_of_edges()}):",
    ]
    for u, v, data in sorted(G.edges(data=True)):
        lines.append(f"    {u:>3} -- {v:<3}  capacity = {data['capacity']:6.1f} Mbps")
    return "\n".join(lines)
