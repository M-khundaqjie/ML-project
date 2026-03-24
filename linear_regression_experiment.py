import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

SEED = 42

ratings  = pd.read_csv("rating.csv")
movies   = pd.read_csv("movie.csv")
g_scores = pd.read_csv("genome_scores.csv")
g_tags   = pd.read_csv("genome_tags.csv")

df = ratings.merge(movies, on="movieId", how="left")
df["liked"] = (df["rating"] >= 4).astype(int)

movies_with_genome = g_scores["movieId"].unique()
df = df[df["movieId"].isin(movies_with_genome)].copy()

SAMPLE_SIZE = 35_000
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=SEED)

df = df.reset_index(drop=True)

genres_dummies = df["genres"].str.get_dummies(sep="|")

genome = g_scores.merge(g_tags, on="tagId", how="left")

genome_pivot = genome.pivot_table(
    index="movieId", columns="tag", values="relevance", aggfunc="first"
)
genome_pivot.columns = [f"genome_{c}" for c in genome_pivot.columns]

genome_features = df[["movieId"]].merge(
    genome_pivot, left_on="movieId", right_index=True, how="left"
)
genome_features = genome_features.drop(columns=["movieId"]).fillna(0)

combined_features = pd.concat(
    [genres_dummies, genome_features],
    axis=1,
)

representations = {
    "Genres Only":   genres_dummies,
    "Genome Only":   genome_features,
    "Genres+Genome": combined_features,
}

y = df["liked"].values

indices = np.arange(len(y))
idx_train, idx_test = train_test_split(
    indices, test_size=0.20, random_state=SEED, stratify=y
)

y_train, y_test = y[idx_train], y[idx_test]

scaled_data = {}
for name, X_full in representations.items():
    X_full_arr = X_full.values.astype(np.float64)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full_arr[idx_train])
    X_te = scaler.transform(X_full_arr[idx_test])
    scaled_data[name] = (X_tr, X_te)

models = {}
for name, (X_tr, X_te) in scaled_data.items():
    model = LinearRegression()
    model.fit(X_tr, y_train)
    models[name] = model

predictions = {}
for name, model in models.items():
    X_te = scaled_data[name][1]
    y_score = model.predict(X_te)
    y_pred  = (y_score >= 0.5).astype(int)
    predictions[name] = {"y_score": y_score, "y_pred": y_pred}

results = []
for name in representations:
    y_pred  = predictions[name]["y_pred"]
    y_score = predictions[name]["y_score"]
    results.append({
        "Representation": name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC":   roc_auc_score(y_test, y_score),
    })

results_df = pd.DataFrame(results).set_index("Representation")

COLORS = ["#4361ee", "#f72585", "#4cc9f0"]

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, color) in enumerate(zip(representations, COLORS)):
    vals = results_df.loc[name, metrics].values
    bars = ax.bar(x + i * width, vals, width, label=name, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Performance — Feature Representation Comparison")
ax.legend(frameon=True, framealpha=0.9)
sns.despine(left=True)
plt.tight_layout()
plt.savefig("lr_plot_A_metrics_comparison.png", bbox_inches="tight")
plt.show()


fig, ax = plt.subplots(figsize=(8, 7))
for name, color in zip(representations, COLORS):
    y_score = predictions[name]["y_score"]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_val = roc_auc_score(y_test, y_score)
    ax.plot(fpr, tpr, color=color, linewidth=2.2,
            label=f"{name}  (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], ls="--", color="grey", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Feature Representation Comparison")
ax.legend(loc="lower right", frameon=True, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig("lr_plot_B_roc_curves.png", bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, color) in zip(axes, zip(representations, COLORS)):
    cm = confusion_matrix(y_test, predictions[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Disliked", "Liked"],
                yticklabels=["Disliked", "Liked"], ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)

fig.suptitle("Confusion Matrices — Feature Representation Comparison",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("lr_plot_C_confusion_matrices.png", bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (name, color) in zip(axes, zip(representations, COLORS)):
    y_score = predictions[name]["y_score"]
    scores_liked    = y_score[y_test == 1]
    scores_disliked = y_score[y_test == 0]

    ax.hist(scores_disliked, bins=40, alpha=0.65, color="#f72585",
            label="Disliked (true)", edgecolor="white", linewidth=0.5)
    ax.hist(scores_liked, bins=40, alpha=0.65, color="#4361ee",
            label="Liked (true)", edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="black", ls="--", linewidth=1.2, label="Threshold (0.5)")
    ax.set_xlabel("Predicted Score")
    ax.set_title(name)
    ax.legend(fontsize=9, frameon=True)

axes[0].set_ylabel("Frequency")
fig.suptitle("Predicted Score Distributions — Liked vs Disliked",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("lr_plot_D_score_distributions.png", bbox_inches="tight")
plt.show()


TOP_N = 20

fig, axes = plt.subplots(1, 3, figsize=(24, 10))
for ax, (name, color) in zip(axes, zip(representations, COLORS)):
    model = models[name]
    feature_names = list(representations[name].columns)
    coefs = pd.Series(model.coef_, index=feature_names)

    top_pos = coefs.nlargest(TOP_N)
    top_neg = coefs.nsmallest(TOP_N)
    top_coefs = pd.concat([top_neg, top_pos])

    bar_colors = ["#f72585" if v < 0 else "#4361ee" for v in top_coefs.values]
    ax.barh(range(len(top_coefs)), top_coefs.values, color=bar_colors,
            edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(top_coefs)))
    ax.set_yticklabels(top_coefs.index, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(f"{name}")

fig.suptitle(f"Top {TOP_N} Positive & Negative Coefficients — Feature Representation Comparison",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("lr_plot_E_coefficients.png", bbox_inches="tight")
plt.show()
