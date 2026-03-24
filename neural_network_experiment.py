import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLORS = ["#4361ee", "#f72585", "#4cc9f0"]

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

combined_features = pd.concat([genres_dummies, genome_features], axis=1)

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
    X_full_arr = X_full.values.astype(np.float32)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_full_arr[idx_train])
    X_te = scaler.transform(X_full_arr[idx_test])
    scaled_data[name] = (X_tr, X_te)


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


EPOCHS = 50
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.15

models = {}
histories = {}

for name in representations:
    X_tr, X_te = scaled_data[name]
    input_dim = X_tr.shape[1]

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = build_model(input_dim)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_tr, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1,
    )

    models[name] = model
    histories[name] = history.history

predictions = {}
for name, model in models.items():
    X_te = scaled_data[name][1]
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    predictions[name] = {"y_prob": y_prob, "y_pred": y_pred}

results = []
for name in representations:
    y_pred = predictions[name]["y_pred"]
    y_prob = predictions[name]["y_prob"]
    results.append({
        "Representation": name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC":   roc_auc_score(y_test, y_prob),
    })

results_df = pd.DataFrame(results).set_index("Representation")

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
ax.set_title("Neural Network Performance — Feature Representation Comparison")
ax.legend(frameon=True, framealpha=0.9)
sns.despine(left=True)
plt.tight_layout()
plt.savefig("nn_plot_A_metrics_comparison.png", bbox_inches="tight")
plt.show()


fig, ax = plt.subplots(figsize=(8, 7))
for name, color in zip(representations, COLORS):
    y_prob = predictions[name]["y_prob"]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2.2,
            label=f"{name}  (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], ls="--", color="grey", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Neural Network Feature Comparison")
ax.legend(loc="lower right", frameon=True, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig("nn_plot_B_roc_curves.png", bbox_inches="tight")
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

fig.suptitle("Confusion Matrices — Neural Network Feature Comparison",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("nn_plot_C_confusion_matrices.png", bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (name, color) in zip(axes, zip(representations, COLORS)):
    y_prob = predictions[name]["y_prob"]
    probs_liked    = y_prob[y_test == 1]
    probs_disliked = y_prob[y_test == 0]

    ax.hist(probs_disliked, bins=40, alpha=0.65, color="#f72585",
            label="Disliked (true)", edgecolor="white", linewidth=0.5)
    ax.hist(probs_liked, bins=40, alpha=0.65, color="#4361ee",
            label="Liked (true)", edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="black", ls="--", linewidth=1.2, label="Threshold (0.5)")
    ax.set_xlabel("Predicted Probability")
    ax.set_title(name)
    ax.legend(fontsize=9, frameon=True)

axes[0].set_ylabel("Frequency")
fig.suptitle("Predicted Probability Distributions — Liked vs Disliked",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("nn_plot_D_probability_distributions.png", bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col")

for col, (name, color) in enumerate(zip(representations, COLORS)):
    hist = histories[name]
    epochs_range = range(1, len(hist["loss"]) + 1)

    ax_loss = axes[0, col]
    ax_loss.plot(epochs_range, hist["loss"],
                 color=color, linewidth=2, label="Train Loss")
    ax_loss.plot(epochs_range, hist["val_loss"],
                 color=color, linewidth=2, linestyle="--", label="Val Loss")
    ax_loss.set_ylabel("Loss" if col == 0 else "")
    ax_loss.set_title(name)
    ax_loss.legend(fontsize=9, frameon=True)
    ax_loss.grid(True, alpha=0.3)

    ax_acc = axes[1, col]
    ax_acc.plot(epochs_range, hist["accuracy"],
                color=color, linewidth=2, label="Train Accuracy")
    ax_acc.plot(epochs_range, hist["val_accuracy"],
                color=color, linewidth=2, linestyle="--", label="Val Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy" if col == 0 else "")
    ax_acc.legend(fontsize=9, frameon=True)
    ax_acc.grid(True, alpha=0.3)

fig.suptitle("Training History — Loss & Accuracy per Feature Representation",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("nn_plot_E_training_history.png", bbox_inches="tight")
plt.show()
