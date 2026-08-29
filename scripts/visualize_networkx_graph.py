"""NetworkX visualization script for 64-node EEG subject graphs and random geometric graphs.

Supports both:
1. Standalone random geometric graph rendering (64 nodes, radius=0.32) saved to `graph_subject.png`.
2. PyTorch Geometric (PyG) EEG subject graph rendering using NetworkX and standard electrode layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def render_random_geometric_graph(
    num_nodes: int = 64,
    radius: float = 0.32,
    output_path: str | Path = "graph_subject.png",
    dpi: int = 600,
) -> None:
    """Generate and save a random geometric graph using NetworkX."""
    G = nx.random_geometric_graph(num_nodes, radius)

    plt.figure(figsize=(8, 8))
    pos = nx.get_node_attributes(G, "pos")

    nx.draw_networkx_nodes(G, pos, node_size=60, node_color="#1f77b4", alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")

    plt.axis("off")
    out_p = Path(output_path)
    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved random geometric graph to {out_p.resolve()} ({dpi} DPI)")


def pyg_to_networkx_graph(pyg_data: Any) -> nx.Graph:
    """Convert a PyTorch Geometric Data object to a NetworkX Graph."""
    G = nx.Graph()
    num_nodes = pyg_data.num_nodes
    edge_index = pyg_data.edge_index.cpu().numpy()

    for i in range(num_nodes):
        G.add_node(i)

    for src, dst in edge_index.T:
        if src < dst:
            G.add_edge(src, dst)

    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render NetworkX 64-node Subject Graph")
    parser.add_argument("--out", type=str, default="graph_subject.png", help="Output PNG path")
    parser.add_argument("--nodes", type=int, default=64, help="Number of nodes (EEG channels)")
    parser.add_argument("--radius", type=float, default=0.32, help="Distance threshold")
    parser.add_argument("--dpi", type=int, default=600, help="DPI resolution")
    args = parser.parse_args()

    render_random_geometric_graph(
        num_nodes=args.nodes,
        radius=args.radius,
        output_path=args.out,
        dpi=args.dpi,
    )
