import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
os.makedirs("data", exist_ok=True)

K_HOPS = int(os.environ.get("K_HOPS", 2))  # how many hops around v to include in the ego network

# ---- load what step 0 built ------------------------------------------------
with open("data/graph.gpickle", "rb") as f:
    G = pickle.load(f)
labels_df = pd.read_csv("data/labels.csv")

print(f"Loaded graph with {G.number_of_nodes()} nodes")
print(f"Building ego networks for {len(labels_df)} instances (k={K_HOPS} hops)...")

# Pre-compute clustering coefficients once (expensive to redo per node)
clustering = nx.clustering(G)


def get_ego_subgraph(G, v, k=K_HOPS):
    """BFS out to k hops from v. Returns the induced subgraph (a small nx.Graph)."""
    nodes_in_range = nx.single_source_shortest_path_length(G, v, cutoff=k)
    sub_nodes = list(nodes_in_range.keys())
    return G.subgraph(sub_nodes).copy(), sub_nodes


def node_feature_vector(G, n, clustering):
    return [
        float(G.nodes[n]["active_at_t0"]),
        float(G.nodes[n]["activity_level"]),
        float(G.nodes[n]["historical_action_freq"]),
        float(np.log1p(G.degree(n))),
        float(clustering.get(n, 0.0)),
    ]


def handcrafted_features(G, v, clustering):
    """The (d) 'Handcrafted Features' block from the PS, for the ego user v."""
    neighbors = list(G.neighbors(v))
    degree = len(neighbors)
    active_neighbors = sum(1 for u in neighbors if G.nodes[u]["active_at_t0"])
    influence_ratio = active_neighbors / degree if degree > 0 else 0.0
    frac_active = influence_ratio  # "fraction of v's neighbours who already performed the action"
    return [
        degree,
        clustering.get(v, 0.0),
        active_neighbors,
        influence_ratio,
        frac_active,
        G.nodes[v]["activity_level"],
        G.nodes[v]["historical_action_freq"],
    ]


instances = []
skipped = 0
for _, row in labels_df.iterrows():
    v = int(row["node"])
    label = int(row["label"])

    sub_G, sub_nodes = get_ego_subgraph(G, v, K_HOPS)
    if sub_G.number_of_nodes() < 2:
        skipped += 1
        continue  # isolated / too-small ego network, not useful

    # map original node ids -> local 0..n-1 ids required by PyG
    local_id = {n: i for i, n in enumerate(sub_nodes)}
    v_local = local_id[v]

    x = torch.tensor([node_feature_vector(G, n, clustering) for n in sub_nodes], dtype=torch.float)

    edges = list(sub_G.edges())
    if len(edges) == 0:
        skipped += 1
        continue
    edge_index = torch.tensor(
        [[local_id[a] for a, b in edges] + [local_id[b] for a, b in edges],
         [local_id[b] for a, b in edges] + [local_id[a] for a, b in edges]],
        dtype=torch.long,
    )  # undirected: add both directions

    handcrafted = torch.tensor(handcrafted_features(G, v, clustering), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.float),
        v_local=torch.tensor([v_local], dtype=torch.long),  # which row of x is the ego user
        handcrafted=handcrafted.unsqueeze(0),  # shape (1, n_handcrafted)
        node_id=torch.tensor([v], dtype=torch.long),
    )
    instances.append(data)

print(f"Built {len(instances)} ego-network instances ({skipped} skipped as too small)")
print(f"Handcrafted feature dim: {instances[0].handcrafted.shape[1]}")
print(f"Node feature dim (for GNN): {instances[0].x.shape[1]}")

torch.save(instances, "data/instances.pt")
torch.save(instances, f"data/instances_k{K_HOPS}.pt")  # keep a copy per k, for sweeps
print("\nSaved: data/instances.pt")
print("Next step: python step2_model.py  (defines the model)")
print("Then:      python step3_train.py  (trains + evaluates + baselines)")
