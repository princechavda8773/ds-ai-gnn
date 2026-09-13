import random
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
os.makedirs("data", exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- 1. Build the social graph -------------------------------------------
N_NODES = 6000          # number of users. Keep small so training is fast (~minutes on CPU).
BA_M = 4                # each new node attaches to 4 existing nodes -> realistic hub structure

G = nx.barabasi_albert_graph(N_NODES, BA_M, seed=RANDOM_SEED)
print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---- 2. Give every user some static attributes ----------------------------
for node in G.nodes():
    G.nodes[node]["activity_level"] = float(np.clip(np.random.beta(2, 5), 0, 1))
    G.nodes[node]["historical_action_freq"] = float(np.clip(np.random.beta(2, 8), 0, 1))

# ---- 3. Simulate an Independent Cascade of "the action" -------------------
def simulate_independent_cascade(G, seed_fraction=0.05, base_prob=0.25, max_steps=15):
    """
    Returns: dict {node: activation_time}. Nodes that never activate are absent.
    A node's chance of being infected by an active neighbour is boosted by
    that node's own `activity_level`, so more "susceptible" users are more
    likely to act -> makes the label learnable from features, not just noise.
    """
    n_seeds = max(1, int(seed_fraction * G.number_of_nodes()))
    seeds = random.sample(list(G.nodes()), n_seeds)

    activation_time = {s: 0 for s in seeds}
    frontier = set(seeds)
    t = 0
    while frontier and t < max_steps:
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v in activation_time:
                    continue  # already active
                susceptibility = G.nodes[v]["activity_level"]
                p = base_prob * (0.5 + susceptibility)  # more active users are easier to influence
                if random.random() < p:
                    activation_time[v] = t + 1
                    next_frontier.add(v)
        frontier = next_frontier
        t += 1
    return activation_time

activation_time = simulate_independent_cascade(G)
print(f"Cascade finished: {len(activation_time)} / {N_NODES} users eventually acted")

# ---- 4. Split into observation time t0 and final time T -------------------
max_t = max(activation_time.values()) if activation_time else 1
T0 = max(1, max_t // 2)   # observation cutoff (what the model is ALLOWED to see)
T_FINAL = max_t            # final outcome (what we're trying to predict)

nx.set_node_attributes(
    G, {n: (activation_time.get(n, 10**9) <= T0) for n in G.nodes()}, "active_at_t0"
)

# ---- 5. Build the label table ---------------------------------------------
# Only keep users who had NOT acted yet at t0 -> that's our prediction set.
rows = []
for node in G.nodes():
    if G.nodes[node]["active_at_t0"]:
        continue  # already acted before observation cutoff -> not a valid instance
    label = int(activation_time.get(node, 10**9) <= T_FINAL and node in activation_time)
    rows.append({"node": node, "label": label})

labels_df = pd.DataFrame(rows)
print(f"\nInstances (users who hadn't acted by t0): {len(labels_df)}")
print(f"Positive rate (eventually acted): {labels_df['label'].mean():.3f}")

# ---- 6. Save everything ----------------------------------------------------
with open("data/graph.gpickle", "wb") as f:
    pickle.dump(G, f)
labels_df.to_csv("data/labels.csv", index=False)
with open("data/meta.pkl", "wb") as f:
    pickle.dump({"T0": T0, "T_FINAL": T_FINAL}, f)

print("\nSaved: data/graph.gpickle, data/labels.csv, data/meta.pkl")
print("Next step: python step1_ego_sampling_and_features.py")
