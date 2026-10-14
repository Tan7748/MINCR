"""
visualization.py
----------------
Draws the network BEFORE and AFTER the intervention.

Links are coloured by utilisation:
    green  = comfortable   (< 60%)
    orange = getting busy  (60% - 80%)
    red    = CONGESTED     (> 80%, our threshold)
Link thickness also grows with utilisation, so congestion is visible even
in a black-and-white printout of the lab record.

We use the "Agg" backend so the script also works on a machine with no
display (a lab PC over SSH, for example). Figures are saved as PNG files.
"""

import matplotlib
matplotlib.use("Agg")           # must be set BEFORE importing pyplot
import matplotlib.pyplot as plt
import networkx as nx

from network import link_key


def _edge_colour(u, threshold):
    if u > threshold:
        return "#d62728"        # red
    if u > 0.60:
        return "#ff7f0e"        # orange
    return "#2ca02c"            # green


def draw_state(G, flows, paths, state, title, filename,
               highlight_flow=None):
    """Draw one network state and save it to `filename`."""
    pos = nx.get_node_attributes(G, "pos")
    threshold = state["threshold"]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    edge_colours, widths, labels = [], [], {}
    for u, v in G.edges():
        util = state["util"][link_key(u, v)]
        edge_colours.append(_edge_colour(util, threshold))
        widths.append(1.5 + 5.0 * min(util, 1.2))
        labels[(u, v)] = (f"{state['loads'][link_key(u, v)]:.0f}/"
                          f"{G[u][v]['capacity']:.0f}\n{util * 100:.0f}%")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colours, width=widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1100,
                           node_color="#dbe9ff", edgecolors="#20304a")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax,
                                 font_size=7, bbox=dict(alpha=0.0))

    # Draw the rerouted flow's path on top, as a dashed blue overlay.
    if highlight_flow and paths.get(highlight_flow):
        path = paths[highlight_flow]
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                               edge_color="#1f4fd8", width=3.0,
                               style="dashed", alpha=0.9)

    subtitle = (f"C(G) = {state['C']:.4f}   |   MLU = {state['MLU'] * 100:.1f}%"
                f"   |   congested links = {len(state['congested'])}"
                f"   (threshold {threshold * 100:.0f}%)")
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.axis("off")

    legend = [
        plt.Line2D([], [], color="#2ca02c", lw=4, label="utilisation < 60%"),
        plt.Line2D([], [], color="#ff7f0e", lw=4, label="60% - 80%"),
        plt.Line2D([], [], color="#d62728", lw=4, label="> 80% (congested)"),
    ]
    if highlight_flow:
        legend.append(plt.Line2D([], [], color="#1f4fd8", lw=3, ls="--",
                                 label=f"rerouted flow {highlight_flow}"))
    ax.legend(handles=legend, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    return filename


def draw_responsibility_bar(scores, ranked, filename, title):
    """Bar chart of the CR scores - the clearest single slide for the demo."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ids = ranked
    values = [scores[f]["CR"] for f in ids]
    colours = ["#d62728" if i == 0 else "#7f9dc4" for i in range(len(ids))]

    ax.bar(ids, values, color=colours)
    ax.set_ylabel("Congestion Responsibility  CR$_i$")
    ax.set_xlabel("flow")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values + [0.01]) * 1.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)
    return filename
