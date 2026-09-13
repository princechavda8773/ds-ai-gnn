import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from model import SocialInfluenceModel, GNNEncoder

import os
os.makedirs("outputs", exist_ok=True)

import os
os.makedirs("outputs", exist_ok=True)

# --- Config, overridable via environment variables (used by the sweep script) ---
GNN_TYPE = os.environ.get("GNN_TYPE", "gcn")            # "gcn" or "gat"
INSTANCES_PATH = os.environ.get("INSTANCES_PATH", "data/instances.pt")
RESULTS_TAG = os.environ.get("RESULTS_TAG", "")          # optional label for results file

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Load + split data
# ---------------------------------------------------------------------------
instances = torch.load(INSTANCES_PATH, weights_only=False)
np.random.shuffle(instances)

n = len(instances)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
train_set = instances[:n_train]
val_set = instances[n_train:n_train + n_val]
test_set = instances[n_train + n_val:]
print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

_hc_train = np.stack([d.handcrafted.squeeze(0).numpy() for d in train_set])
_hc_mean = _hc_train.mean(axis=0)
_hc_std = _hc_train.std(axis=0) + 1e-6
for d in instances:
    d.handcrafted = ((d.handcrafted - torch.tensor(_hc_mean, dtype=torch.float))
                      / torch.tensor(_hc_std, dtype=torch.float))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)


# ---------------------------------------------------------------------------
# Generic train/eval helpers (reused for the full model AND ablations)
# ---------------------------------------------------------------------------
def train_gnn_model(model, train_loader, val_loader, epochs=60, lr=0.002, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # pos_weight up-weights the rare positive class (our label is ~9-11% positive)
    pos_weight = torch.tensor([4.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_loss, best_state, patience, bad_epochs = float("inf"), None, 15, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        val_loss = compute_val_loss(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (epoch + 1) % 5 == 0:
            val_metrics = evaluate_gnn_model(model, val_loader)
            print(f"  epoch {epoch+1:3d} | train_loss {total_loss/len(train_loader.dataset):.4f} "
                  f"| val_loss {val_loss:.4f} | val_auc {val_metrics['auc']:.4f}")

        if bad_epochs >= patience:  # early stopping
            if verbose:
                print(f"  early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def compute_val_loss(model, loader, criterion):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        logits = model(batch)
        loss = criterion(logits, batch.y)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / count


@torch.no_grad()
def evaluate_gnn_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logits = model(batch)
        probs = torch.sigmoid(logits)
        all_probs.append(probs)
        all_labels.append(batch.y)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def print_metrics(name, m):
    print(f"{name:32s} | AUC {m['auc']:.3f} | F1 {m['f1']:.3f} "
          f"| Precision {m['precision']:.3f} | Recall {m['recall']:.3f}")


results = {}

# ---------------------------------------------------------------------------
# 2. FULL MODEL: GNN embedding + handcrafted features (the proposed model)
# ---------------------------------------------------------------------------
print(f"\n=== Training FULL MODEL (GNN + handcrafted features) [gnn_type={GNN_TYPE}] ===")
node_feat_dim = instances[0].x.shape[1]
handcrafted_dim = instances[0].handcrafted.shape[1]

full_model = SocialInfluenceModel(node_feat_dim, handcrafted_dim, hidden_dim=16, gnn_type=GNN_TYPE)
full_model = train_gnn_model(full_model, train_loader, val_loader, epochs=100)
results["Full model (GNN + handcrafted)"] = evaluate_gnn_model(full_model, test_loader)
torch.save(full_model.state_dict(), "outputs/full_model.pt")

# ---------------------------------------------------------------------------
# 3. BASELINE A: Logistic Regression on handcrafted features only
# ---------------------------------------------------------------------------
print("\n=== Baseline: Logistic Regression (handcrafted features only) ===")


def to_numpy_handcrafted(dataset):
    X = np.stack([d.handcrafted.squeeze(0).numpy() for d in dataset])
    y = np.array([d.y.item() for d in dataset])
    return X, y


X_train, y_train = to_numpy_handcrafted(train_set)
X_test, y_test = to_numpy_handcrafted(test_set)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
logreg.fit(X_train_s, y_train)
probs = logreg.predict_proba(X_test_s)[:, 1]
preds = (probs > 0.5).astype(int)
results["Logistic Regression (handcrafted only)"] = {
    "auc": roc_auc_score(y_test, probs),
    "f1": f1_score(y_test, preds, zero_division=0),
    "precision": precision_score(y_test, preds, zero_division=0),
    "recall": recall_score(y_test, preds, zero_division=0),
}

# ---------------------------------------------------------------------------
# 4. BASELINE B: Node2Vec-style embeddings + MLP
# ---------------------------------------------------------------------------
print("\n=== Baseline: Node2Vec-style embeddings + MLP ===")
import pickle
import random as _random

with open("data/graph.gpickle", "rb") as f:
    import networkx as nx
    G = pickle.load(f)

node_list = sorted(G.nodes())


def random_walks(G, num_walks=10, walk_length=20, seed=42):
    rng = _random.Random(seed)
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(walk_length - 1):
                nbrs = list(G.neighbors(walk[-1]))
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append([str(n) for n in walk])
    return walks


walks = random_walks(G, num_walks=8, walk_length=15)
print(f"  generated {len(walks)} random walks, training skip-gram embeddings...")

from gensim.models import Word2Vec  # lightweight, pure-Python-friendly skip-gram trainer

w2v = Word2Vec(sentences=walks, vector_size=32, window=5, min_count=0, sg=1,
               workers=1, epochs=5, seed=42)


def to_n2v_features(dataset):
    X = np.stack([w2v.wv[str(d.node_id.item())] for d in dataset])
    y = np.array([d.y.item() for d in dataset])
    return X, y


Xn_train, yn_train = to_n2v_features(train_set)
Xn_test, yn_test = to_n2v_features(test_set)

mlp_n2v = LogisticRegression(class_weight="balanced", max_iter=1000)  # "+ MLP classifier" -> a
# simple linear/MLP classifier head on top of frozen node2vec embeddings, as in the PS table.
mlp_n2v.fit(Xn_train, yn_train)
probs = mlp_n2v.predict_proba(Xn_test)[:, 1]
preds = (probs > 0.5).astype(int)
results["Node2Vec + MLP"] = {
    "auc": roc_auc_score(yn_test, probs),
    "f1": f1_score(yn_test, preds, zero_division=0),
    "precision": precision_score(yn_test, preds, zero_division=0),
    "recall": recall_score(yn_test, preds, zero_division=0),
    "accuracy": accuracy_score(yn_test, preds),
}

# ---------------------------------------------------------------------------
# 5. BASELINE C: Plain GCN, no handcrafted-feature fusion
# ---------------------------------------------------------------------------
print("\n=== Baseline: Plain GCN (no handcrafted features) ===")


class PlainGCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=16):
        super().__init__()
        self.encoder = GNNEncoder(node_feat_dim, hidden_dim, gnn_type="gcn")
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch):
        node_embeddings = self.encoder(batch.x, batch.edge_index)
        ego_embeddings = node_embeddings[batch.ptr[:-1] + batch.v_local]
        return self.head(ego_embeddings).squeeze(-1)


plain_gcn = PlainGCN(node_feat_dim, hidden_dim=16)
plain_gcn = train_gnn_model(plain_gcn, train_loader, val_loader, epochs=100, verbose=False)
results["Plain GCN (no handcrafted features)"] = evaluate_gnn_model(plain_gcn, test_loader)

# ---------------------------------------------------------------------------
# 6. ABLATIONS required in Section 5 of the PS
# ---------------------------------------------------------------------------
print("\n=== Ablation: WITHOUT instance normalization (LayerNorm) ===")


class GNNEncoderNoNorm(nn.Module):
    """Same as GNNEncoder but skips the LayerNorm step, to isolate its effect."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return h


class FullModelNoNorm(nn.Module):
    def __init__(self, node_feat_dim, handcrafted_dim, hidden_dim=16):
        super().__init__()
        self.encoder = GNNEncoderNoNorm(node_feat_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + handcrafted_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        node_embeddings = self.encoder(batch.x, batch.edge_index)
        ego_embeddings = node_embeddings[batch.ptr[:-1] + batch.v_local]
        fused = torch.cat([ego_embeddings, batch.handcrafted], dim=1)
        return self.head(fused).squeeze(-1)


no_norm_model = FullModelNoNorm(node_feat_dim, handcrafted_dim, hidden_dim=16)
no_norm_model = train_gnn_model(no_norm_model, train_loader, val_loader, epochs=100, verbose=False)
results["Ablation: Full model WITHOUT instance norm"] = evaluate_gnn_model(no_norm_model, test_loader)

# 'without handcrafted features' ablation = exactly the "Plain GCN" baseline above,
# already computed and included in results.
results["Ablation: Full model WITHOUT handcrafted features"] = results["Plain GCN (no handcrafted features)"]

# ---------------------------------------------------------------------------
# 7. Final report
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL RESULTS (test set)")
print("=" * 90)
for name, m in results.items():
    print_metrics(name, m)

# Save results to CSV so sweep scripts can aggregate across runs
import csv
results_file = "outputs/results_log.csv"
file_exists = os.path.exists(results_file)
with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["tag", "gnn_type", "instances_path", "model", "auc", "f1", "precision", "recall"])
    for name, m in results.items():
        writer.writerow([RESULTS_TAG, GNN_TYPE, INSTANCES_PATH, name,
                          f"{m['auc']:.4f}", f"{m['f1']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}"])
print(f"\nAppended results to {results_file}")

print("Saved trained full-model weights -> outputs/full_model.pt")
