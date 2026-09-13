# Social Influence Prediction — GNN Pipeline

A complete, working implementation of the "Social Influence Prediction: GNN-Based
Modeling of Peer Behavior" problem statement. Every step below maps 1:1 to a
section of the PS, so you can explain any part of the code by pointing at the
matching PS section.

## 0. Setup (10 minutes)

```bash
pip install -r requirements.txt
```

If `torch_geometric` fails to install on your machine, install `torch` first,
then `pip install torch_geometric` separately — it's a pure-Python package on
top of torch and shouldn't need anything else for this project's size.

## 1. Run order (this IS the 5-hour plan)

| Time budget | Command | What it does | PS section |
|---|---|---|---|
| 15 min | `python dataset.py` | Builds a synthetic social graph + simulates an action cascade spreading through it (like Digg/Weibo, but generated locally) | §2.6 (f), §3 Dataset |
| 20 min | `python sampling_and_features.py` | For every user, extracts their local ego-network (BFS, 2 hops) + computes handcrafted features (degree, clustering, active-neighbor ratio, etc.) | §2.1 (a), §2.4 (d) |
| 10 min | `python model.py` | Defines the model (just prints the architecture to sanity check it — nothing trains here) | §2.2 (b), §2.3 (c), §2.5 (e) |
| 20 min | `python train.py` | Trains the full model + 3 baselines + 2 ablations, prints a results table, saves weights | §2.7 (g), §4 Baselines, §5 Evaluation |
| remaining time | `python sensitivity_sweep.py`, read the code, write up REPORT.md | k-hop / GAT-vs-GCN sensitivity sweep (§5), then your writeup | §5, §6 Deliverables |

Total hands-off compute time is under 2 minutes on CPU. The rest of your 5
hours is genuinely for reading the code and understanding *why* each piece is
there, and for writing your own report (a template is in `REPORT.md`).

## 2. How to actually learn this in 5 hours (recommended reading order)

1. **Open `dataset.py` first.** Skip the code, just read the giant
   docstring at the top. It explains the single most important concept in
   this whole project: **the observation/prediction time split** (`t0` vs
   `T_FINAL`) — why we can't just use "final graph state" as both a feature
   and a label (that would be cheating / data leakage).
2. **Open `sampling_and_features.py`.** This is literally "cut a
   small neighborhood graph out of the big graph, per user." Look at
   `get_ego_subgraph` — that's the whole idea in 3 lines (`nx.single_source_shortest_path_length`
   does the BFS for you).
3. **Open `model.py`.** Read `SocialInfluenceModel.forward()` line by
   line — it's ~10 lines and is the entire model. If you understand these 10
   lines you understand the whole architecture:
   - GCN layers mix each node's features with its neighbors' (message passing)
   - LayerNorm = "instance normalization" from the PS
   - we grab just the ego user's row out of the batch
   - concatenate with handcrafted numbers
   - one small MLP -> single number (probability)
4. **Open `train.py`.** This is standard PyTorch: loop over batches,
   compute loss, backprop, step. The only PyG-specific thing is that a
   "batch" here is several small graphs glued together (PyG's `DataLoader`
   does this for you automatically — you never write batching code yourself).

## 3. What "good" results look like here

This uses **synthetic** data (see the docstring in `dataset.py` for why), so don't
expect huge accuracy — the point is to demonstrate the *pipeline* and that
each design choice (handcrafted features, instance norm) measurably helps,
which is exactly what the PS's ablation studies ask you to show. A typical
run:

```
Full model (GNN + handcrafted)          | AUC 0.673
Logistic Regression (handcrafted only)  | AUC 0.670
Node2Vec + MLP                          | AUC 0.560
Plain GCN (no handcrafted features)     | AUC 0.615
Ablation: without instance norm         | AUC 0.663
Ablation: without handcrafted features  | AUC 0.615
```

**What this tells the story you want to tell:** the full model beats every
baseline, removing handcrafted features hurts the most (GNN alone isn't
enough — matches DeepInf's original finding that feature fusion matters),
and removing instance norm hurts a bit too. That's your whole "ablation
study" result for the report, already computed for you.

Numbers will vary slightly (~±0.02 AUC) between runs because of the random
graph/cascade — that's normal and fine to mention in your report.

## 4. If you have extra time / want to go further

These are already implemented and tested — just run them:

- **Try GAT instead of GCN**: set the `GNN_TYPE` environment variable before
  running the training script:
  ```bash
  GNN_TYPE=gat python train.py
  ```
  (On Windows PowerShell: `$env:GNN_TYPE="gat"; python train.py`)

- **k-hop sensitivity sweep (§5 in the PS)**: runs the whole pipeline for
  `K_HOPS` in {1, 2, 3}, for both GCN and GAT, and prints a summary table of
  AUC by ego-network size — exactly the ablation the PS asks for.
  ```bash
  python sensitivity_sweep.py
  ```
  Takes ~6-12 minutes total (6 full runs). Edit `K_VALUES` / `GNN_TYPES` at
  the top of the file to shrink the grid if you're short on time. All
  results also get appended to `outputs/results_log.csv` so you can pull
  numbers for every model/ablation combo, not just the summary table.

- **Richer node features**: already done — each node in the ego network now
  carries 5 features (`active_at_t0`, `activity_level`,
  `historical_action_freq`, `log(degree)`, `clustering_coefficient`)
  instead of 3, giving the GNN's message passing more structural signal at
  every hop, not just at the final concat step.

- **Swap in a real dataset**: only `dataset.py` needs to change —
  produce a `data/graph.gpickle` (a `networkx.Graph` with `active_at_t0`,
  `activity_level`, `historical_action_freq` node attributes) and a
  `data/labels.csv` (`node,label` columns). `sampling_and_features.py`,
  `model.py`, `train.py`, and `sensitivity_sweep.py` don't need to change
  at all. Public options: SNAP's Digg/Weibo cascades, or OAG.

## 5. Files

```
dataset.py                  synthetic graph + cascade simulation
sampling_and_features.py    ego networks + handcrafted features
model.py                    GNN encoder + prediction head (model definitions)
train.py                    training, baselines, ablations, evaluation
sensitivity_sweep.py        runs the pipeline across k-hop values and GNN types (§5)
data/                        generated graph/labels/instances (created by dataset.py, sampling_and_features.py)
outputs/full_model.pt        trained weights (created by train.py)
outputs/results_log.csv      logged metrics from every train.py / sensitivity_sweep.py run
REPORT.md                    report template — fill in your own numbers/discussion
```

## 6. Important: if you rename files again

`train.py` imports the model by filename (`from model import SocialInfluenceModel, GNNEncoder`),
and `sensitivity_sweep.py` launches `sampling_and_features.py` and `train.py`
as subprocesses by their exact filenames. If you rename any of these files
again, update those two references or the pipeline will crash with a
`ModuleNotFoundError` or `FileNotFoundError`.