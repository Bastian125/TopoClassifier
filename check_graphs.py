import os
import torch
from torch_geometric.data import Data


def check_graph_file(file_path):
    if not os.path.exists(file_path):
        print(f"[✗] File not found: {file_path}")
        return

    try:
        graphs = torch.load(file_path)
    except Exception as e:
        print(f"[✗] Failed to load file: {file_path}")
        print(f"    Error: {e}")
        return

    if not isinstance(graphs, list):
        print(f"[✗] Expected list of graphs, got {type(graphs)}")
        return

    for i, g in enumerate(graphs[:5]):  # Check first 5 graphs for brevity
        if not isinstance(g, Data):
            print(f"[✗] Graph {i} is not a torch_geometric.data.Data object.")
            continue

        for attr in ["x", "y", "edge_index", "weights"]:
            if not hasattr(g, attr):
                print(f"[✗] Graph {i} is missing attribute: {attr}")
                continue

        if g.x.shape[0] != g.y.shape[0] or g.x.shape[0] != g.weights.shape[0]:
            print(f"[✗] Graph {i} has inconsistent node/label/weight sizes.")
            continue

        if g.edge_index.ndim != 2 or g.edge_index.shape[0] != 2:
            print(f"[✗] Graph {i} has invalid edge_index shape: {g.edge_index.shape}")
            continue

        print(
            f"[✓] Graph {i} passed checks with {g.x.shape[0]} nodes and {g.edge_index.shape[1]} edges."
        )

    print(
        f"\n[✓] Checked {min(5, len(graphs))} of {len(graphs)} graphs in: {file_path}"
    )


# Example usage
if __name__ == "__main__":
    graph_file = "/ceph/e4/users/bschuchardt/public/MA/data/mc20e_graphs_train_chunk0.pt"
    check_graph_file(graph_file)
