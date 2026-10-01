# ============================================================
# Statistical robustness checks requested by Reviewer 1 and
# Reviewer 2 -- run this AFTER (or instead of, doesn't matter)
# the mixed-dialect probe script, in the same Colab notebook.
#
# PART A -- Reviewer 1 asked for:
#   "Silhouette scores... or statistical permutation tests"
#   to back up the claim that the 3 dialects are separated in
#   the embedding space, instead of just showing t-SNE pictures.
#
#   This uses the ORIGINAL exploratory embedding model
#   (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
#   the SAME one used for the centroid-distance / t-SNE analysis
#   in Sections III-A/III-B of the paper. It does NOT touch the
#   MARBERT classifier -- this is a separate, honest check on the
#   separate exploratory analysis, matching the scope split we
#   already added to the paper.
#
#   We compute:
#     1. The silhouette score for the 3 dialect labels in the
#        50-dim PCA space (ranges -1 to +1; higher = better
#        separated clusters; 0 = no real separation; this is a
#        real, standard clustering-quality statistic, not just
#        a picture).
#     2. A permutation test: we shuffle the dialect labels
#        randomly 1,000 times and recompute the average pairwise
#        centroid distance each time, building a "what if the
#        labels were random" baseline. We then check: is the
#        REAL average centroid distance bigger than almost all of
#        the 1,000 random-shuffle versions? If yes, that's real
#        statistical evidence the separation is not due to chance.
#
# PART B -- Reviewer 2 asked for:
#   "train a baseline dialect classifier... using
#   cross-validation and report macro-F1 and confusion matrices"
#
#   We already did this with ONE train/test split (v6). This part
#   repeats it properly with 5-FOLD CROSS-VALIDATION: the data is
#   split into 5 chunks, we train on 4 chunks and test on the 5th,
#   five times (each chunk gets to be the test set once), and we
#   report the AVERAGE accuracy and macro-F1 across all 5 runs,
#   plus how much they varied. This is the standard way reviewers
#   expect "cross-validation" to be done, instead of one lucky (or
#   unlucky) single split.
#
#   This uses MARBERT (same as v6, same anisotropic premise +
#   defuzzification model) since that's the classifier Section
#   III-C reports on.
# ============================================================

!pip install -q transformers sentence-transformers

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import silhouette_score, accuracy_score, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---- Load data ----
DATA_PATH = "Copy of ARABIC_DIALECTS_ANFIS_GOLD (1)(1) (1).xlsx"   # <-- check this matches your Colab Files panel
df = pd.read_excel(DATA_PATH, sheet_name="GOLD_Long_UTT_RESP")
labels = df["dialect_name"].values
dialects = ["Levantine", "Egyptian", "Gulf"]

# ============================================================
# PART A -- silhouette score + permutation test on the
#           ORIGINAL exploratory embedding space (MiniLM)
# ============================================================
print("\n" + "=" * 70)
print("PART A: Silhouette score + permutation test (exploratory embedding space)")
print("=" * 70)

print("\nLoading sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ...")
minilm = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

print("Embedding all utterances (this is a small/fast model, should be quick)...")
utter_emb_minilm = minilm.encode(df["utterance"].astype(str).tolist(), batch_size=64, show_progress_bar=True)

pca_explore = PCA(n_components=50, random_state=RANDOM_SEED)
z_explore = pca_explore.fit_transform(utter_emb_minilm)
print("PCA explained variance (exploratory, MiniLM space):", pca_explore.explained_variance_ratio_.sum())

# --- 1. Silhouette score for the 3 dialect labels ---
sil = silhouette_score(z_explore, labels)
print(f"\nSilhouette score for the 3 dialect labels in this 50-dim PCA space: {sil:.4f}")
print("(Ranges from -1 to +1. Above 0 means the dialects form somewhat real clusters;")
print(" closer to 0 means heavy overlap; this is the actual number, not a picture.)")

# --- 2. Permutation test on average pairwise centroid distance ---
def avg_pairwise_centroid_distance(z, labels, dialects):
    centroids = {d: z[labels == d].mean(axis=0) for d in dialects}
    dists = []
    for i in range(len(dialects)):
        for j in range(i + 1, len(dialects)):
            d = np.linalg.norm(centroids[dialects[i]] - centroids[dialects[j]])
            dists.append(d)
    return np.mean(dists)

observed_stat = avg_pairwise_centroid_distance(z_explore, labels, dialects)
print(f"\nObserved average pairwise centroid distance (real labels): {observed_stat:.4f}")

N_PERMUTATIONS = 1000
rng = np.random.RandomState(RANDOM_SEED)
null_stats = np.zeros(N_PERMUTATIONS)
shuffled_labels = labels.copy()
for p in range(N_PERMUTATIONS):
    rng.shuffle(shuffled_labels)
    null_stats[p] = avg_pairwise_centroid_distance(z_explore, shuffled_labels, dialects)

p_value = (np.sum(null_stats >= observed_stat) + 1) / (N_PERMUTATIONS + 1)
print(f"\nPermutation test ({N_PERMUTATIONS} random label shuffles):")
print(f"  Average centroid distance under random shuffling: mean={null_stats.mean():.4f}, "
      f"std={null_stats.std():.4f}, max={null_stats.max():.4f}")
print(f"  p-value (chance of seeing distance >= observed, under random labels): {p_value:.4f}")
print("  (A small p-value, e.g. < 0.01, means the real dialect separation is NOT")
print("   something you'd see just from randomly shuffled labels -- i.e. it's real signal.)")

# ============================================================
# PART B -- 5-fold cross-validation of the ANFIS classifier
#           (MARBERT space, same model as v6)
# ============================================================
print("\n" + "=" * 70)
print("PART B: 5-fold cross-validation of the trained ANFIS classifier (MARBERT space)")
print("=" * 70)

print("\nLoading MARBERT...")
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
bert_model = AutoModel.from_pretrained("UBC-NLP/MARBERT").to(DEVICE)
bert_model.eval()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts_ = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts_


@torch.no_grad()
def embed_marbert(texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
        out = bert_model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(pooled.cpu().numpy())
    return np.vstack(all_vecs)


print("Embedding all utterances with MARBERT (this can take a few minutes)...")
utter_emb_marbert = embed_marbert(df["utterance"].astype(str).tolist())


def one_hot(y, dialects):
    return np.array([[1.0 if lab == d else 0.0 for d in dialects] for lab in y])


def anfis_membership_anisotropic(z, centroids, variances, dialects):
    log_w = []
    for d in dialects:
        var_d = variances[d]
        diff2 = (z - centroids[d]) ** 2
        ll = -0.5 * np.sum(np.log(var_d)) - np.sum(diff2 / (2 * var_d), axis=1)
        log_w.append(ll)
    log_w = np.array(log_w)
    log_w -= log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w)
    w_sum = w.sum(axis=0, keepdims=True)
    mu = w / np.clip(w_sum, 1e-12, None)
    return mu.T


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
fold_accuracies = []
fold_macro_f1s = []

for fold_i, (train_idx, test_idx) in enumerate(skf.split(utter_emb_marbert, labels), start=1):
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    z_train = pca.fit_transform(utter_emb_marbert[train_idx])
    z_test = pca.transform(utter_emb_marbert[test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    global_var = z_train.var(axis=0)
    VAR_FLOOR_FRACTION = 0.05
    centroids, variances = {}, {}
    for d in dialects:
        mask = y_train == d
        centroids[d] = z_train[mask].mean(axis=0)
        per_dim_var = z_train[mask].var(axis=0)
        variances[d] = np.maximum(per_dim_var, VAR_FLOOR_FRACTION * global_var)

    mu_train = anfis_membership_anisotropic(z_train, centroids, variances, dialects)
    mu_test = anfis_membership_anisotropic(z_test, centroids, variances, dialects)

    y_train_oh = one_hot(y_train, dialects)
    R, *_ = np.linalg.lstsq(mu_train, y_train_oh, rcond=None)

    crisp_test = mu_test @ R
    pred = np.array(dialects)[crisp_test.argmax(axis=1)]

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, labels=dialects, average="macro")
    fold_accuracies.append(acc)
    fold_macro_f1s.append(macro_f1)
    print(f"Fold {fold_i}: accuracy = {acc:.4f}, macro-F1 = {macro_f1:.4f}")

fold_accuracies = np.array(fold_accuracies)
fold_macro_f1s = np.array(fold_macro_f1s)

print("\n=== 5-fold cross-validation summary ===")
print(f"Accuracy: mean = {fold_accuracies.mean():.4f}, std = {fold_accuracies.std():.4f}")
print(f"Macro-F1: mean = {fold_macro_f1s.mean():.4f}, std = {fold_macro_f1s.std():.4f}")
print("(Compare mean accuracy here to the single-split result already in the paper: 0.8533)")

print("\nDone. Copy this whole block of output back to Claude, including any errors.")
