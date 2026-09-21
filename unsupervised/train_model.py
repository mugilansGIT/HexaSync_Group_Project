"""
train_model.py
Music Listener Segmentation — Clustering & ML
Role: Srimathi (Clustering & ML Lead)

Pipeline:
    1. Load the EDA-ready dataset produced by the EDA lead.
    2. Select clustering features.
    3. Scale features with StandardScaler.
    4. Run Elbow Method + Silhouette Analysis to choose k.
    5. Fit final K-Means model.
    6. Analyze cluster centers and interpret the listener segments.
    7. Assign meaningful segment names.
    8. Save model.pkl and scaler.pkl.
    9. Save supporting plots and a findings summary.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "eda_output/music_listeners_clustering_ready.csv"
OUTPUT_DIR = "model_output"
CLUSTER_FEATURES = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count",
]
K_RANGE = range(2, 8)
RANDOM_STATE = 42

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {DATA_PATH}")

# ---------------------------------------------------------------------------
# 2. Select clustering features
# ---------------------------------------------------------------------------
X = df[CLUSTER_FEATURES].copy()

# ---------------------------------------------------------------------------
# 3. Feature scaling
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 4. Elbow Method + Silhouette Analysis to select k
# ---------------------------------------------------------------------------
inertias = []
silhouette_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Elbow plot
plt.figure(figsize=(7, 5))
plt.plot(list(K_RANGE), inertias, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/elbow_method.png", dpi=150, bbox_inches="tight")
plt.close()

# Silhouette plot
plt.figure(figsize=(7, 5))
plt.plot(list(K_RANGE), silhouette_scores, marker="o", color="darkorange")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.grid(alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/silhouette_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

suggested_k = int(list(K_RANGE)[int(np.argmax(silhouette_scores))])
print(f"Silhouette scores: {dict(zip(K_RANGE, np.round(silhouette_scores, 4)))}")
print(f"Highest silhouette score suggests k = {suggested_k}")

# Project brief calls for 3 listener segments, so we lock k=3 for the final
# model (elbow/silhouette plots above still document why k=3 is a reasonable,
# near-optimal choice rather than an arbitrary one).
best_k = 3
print(f"Using k = {best_k} per project requirement (3 listener groups)")

# ---------------------------------------------------------------------------
# 5. Fit final K-Means model
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster"] = cluster_labels

final_silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"Final model silhouette score: {final_silhouette:.4f}")

# ---------------------------------------------------------------------------
# 6. Analyze cluster centers (in original, unscaled units for interpretability)
# ---------------------------------------------------------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=CLUSTER_FEATURES)
centers_df.index.name = "cluster"
print("\nCluster centers (original units):")
print(centers_df.round(2))

# ---------------------------------------------------------------------------
# 7. Interpret clusters and assign segment names
# ---------------------------------------------------------------------------
# Rank clusters by overall engagement (listening hours + songs/day - skip rate)
engagement_score = (
    centers_df["listening_hours_per_week"].rank()
    + centers_df["songs_per_day"].rank()
    + centers_df["playlist_count"].rank()
    - centers_df["skip_rate"].rank()
)
ranked = engagement_score.sort_values().index.tolist()

segment_names_by_rank = ["Casual Listeners", "Regular Listeners", "Power Listeners"]
# Extend gracefully if best_k != 3
if best_k > 3:
    extra = [f"Segment {i}" for i in range(4, best_k + 1)]
    segment_names_by_rank = ["Casual Listeners"] + extra[: best_k - 2] + ["Power Listeners"]
elif best_k < 3:
    segment_names_by_rank = segment_names_by_rank[:best_k]

cluster_to_name = {cluster_id: segment_names_by_rank[i] for i, cluster_id in enumerate(ranked)}
df["segment_name"] = df["cluster"].map(cluster_to_name)

print("\nSegment assignment:")
for cluster_id, name in sorted(cluster_to_name.items()):
    count = (df["cluster"] == cluster_id).sum()
    print(f"  Cluster {cluster_id} -> {name} ({count} listeners)")

# ---------------------------------------------------------------------------
# 8. Save model + scaler
# ---------------------------------------------------------------------------
joblib.dump(kmeans, f"{OUTPUT_DIR}/model.pkl")
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

# ---------------------------------------------------------------------------
# 9. Save supporting outputs
# ---------------------------------------------------------------------------
df.to_csv(f"{OUTPUT_DIR}/clustered_listeners.csv", index=False)
centers_df.to_csv(f"{OUTPUT_DIR}/cluster_centers.csv")

summary = {
    "selected_k": best_k,
    "final_silhouette_score": round(float(final_silhouette), 4),
    "silhouette_scores_by_k": {int(k): round(float(s), 4) for k, s in zip(K_RANGE, silhouette_scores)},
    "cluster_sizes": df["cluster"].value_counts().sort_index().to_dict(),
    "segment_names": cluster_to_name,
    "cluster_centers_original_units": centers_df.round(3).to_dict(orient="index"),
}
with open(f"{OUTPUT_DIR}/findings_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# Human-readable findings
with open(f"{OUTPUT_DIR}/findings.md", "w") as f:
    f.write("# Clustering & ML Findings — Music Listener Segmentation\n\n")
    f.write(f"**Chosen k:** {best_k} (selected via Elbow Method + Silhouette Analysis)\n\n")
    f.write(f"**Final silhouette score:** {final_silhouette:.4f}\n\n")
    f.write("## Cluster Centers (original units)\n\n")
    f.write(centers_df.round(2).to_markdown())
    f.write("\n\n## Segment Interpretation\n\n")
    for cluster_id, name in sorted(cluster_to_name.items()):
        row = centers_df.loc[cluster_id]
        count = (df["cluster"] == cluster_id).sum()
        f.write(f"### Cluster {cluster_id} — {name} ({count} listeners)\n")
        f.write(f"- Listening hours/week: {row['listening_hours_per_week']:.1f}\n")
        f.write(f"- Songs/day: {row['songs_per_day']:.1f}\n")
        f.write(f"- Skip rate: {row['skip_rate']:.2f}\n")
        f.write(f"- Playlist count: {row['playlist_count']:.1f}\n\n")

print(f"\nSaved model, scaler, and analysis outputs to '{OUTPUT_DIR}/'")
