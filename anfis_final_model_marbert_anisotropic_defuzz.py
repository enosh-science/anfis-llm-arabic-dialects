# ============================================================
# ANFIS layer for Arabic dialect membership inference — v6
# ------------------------------------------------------------
# WHAT'S NEW: a proper DEFUZZIFICATION layer before the decision.
#
# Every version so far (v1-v5) skipped a step. Canonical ANFIS
# (Jang's original architecture) has 5 layers:
#   1) Fuzzification    -- Gaussian membership degree per rule
#   2) Rule firing       -- raw firing strength per rule
#   3) Normalization     -- firing strengths normalized to sum to 1
#                            (this is exactly "mu" in every prior version)
#   4) Defuzzification   -- each rule's CONSEQUENT combined with the
#                            normalized firing strengths -> crisp output
#   5) Decision           -- taken from the crisp output
#
# v1-v5 all jumped straight from step 3 to step 5 -- they argmax'd
# the normalized membership degrees directly, with no consequent
# layer at all. That's a common simplification, but it is not
# actually what "zero-order Sugeno ANFIS" is supposed to do, and
# it's worth fixing properly rather than leaving it implicit.
#
# The fix (standard zero-order Sugeno / Jang's hybrid-learning
# consequent step): each rule d (dialect) gets a constant
# consequent VECTOR r_d in R^3 (one weight per output class). The
# defuzzified crisp output for a sample is the normalized-firing-
# -strength-WEIGHTED SUM of these vectors:
#       y_hat = sum_d  mu_d(z) * r_d
# Stacking the r_d as rows of a 3x3 matrix R, this is just
#       y_hat = mu @ R
# R is fit with ordinary least squares against one-hot training
# labels:
#       R = argmin_R  || one_hot(y_train) - mu_train @ R ||^2
# fit on TRAINING data only (same rule as every other parameter
# in this pipeline) -- so this is still a fully honest, non-
# circular held-out evaluation. It is a genuinely extra learned
# step (a linear layer on top of the fuzzy features), not a
# cosmetic relabeling, so it can really change the result.
#
# This script runs the full comparison on the MARBERT+PCA space
# (since that's the space that matters right now):
#   (a) isotropic premise,  decision WITHOUT defuzzification (= v4's 0.65)
#   (b) isotropic premise,  decision WITH    defuzzification (NEW)
#   (c) anisotropic premise, decision WITHOUT defuzzification (= v5's result)
#   (d) anisotropic premise, decision WITH    defuzzification (NEW)
# all on the identical train/test split, so the four numbers are
# directly comparable.
#
# Same rigor as every version before:
#   - real 80/20 stratified train/test split (identical seed)
#   - PCA fit on training data only
#   - premise parameters (centroids/variances) fit on training data only
#   - consequent matrix R fit on training data only (NEW)
#   - evaluation strictly on the held-out test set
# ============================================================

!pip install -q transformers

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
if DEVICE == "cpu":
    print("Tip: Runtime > Change runtime type > T4 GPU will make this much faster.")

# ---- 1. Load data ----
DATA_PATH = "Copy of ARABIC_DIALECTS_ANFIS_GOLD (1)(1) (1).xlsx"   # <-- check this matches your Colab Files panel
df = pd.read_excel(DATA_PATH, sheet_name="GOLD_Long_UTT_RESP")

# ---- 2. Load MARBERT ----
model_name = "UBC-NLP/MARBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(DEVICE)
bert_model.eval()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_marbert(texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(DEVICE)
        out = bert_model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(pooled.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {min(i + len(batch), len(texts))}/{len(texts)}")
    return np.vstack(all_vecs)


print("Embedding utterances with MARBERT (this can take a few minutes)...")
utter_emb = embed_marbert(df["utterance"].astype(str).tolist())
print("MARBERT embedding shape:", utter_emb.shape)

# ---- 3. Train/test split (identical to v4/v5) ----
labels = df["dialect_name"].values
idx = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
)

# ---- 4. PCA fit on TRAIN ONLY ----
pca = PCA(n_components=50, random_state=RANDOM_SEED)
z_train = pca.fit_transform(utter_emb[train_idx])
z_test = pca.transform(utter_emb[test_idx])
y_train = labels[train_idx]
y_test = labels[test_idx]
D = z_train.shape[1]
print("PCA explained variance (train, MARBERT space):", pca.explained_variance_ratio_.sum())

dialects = ["Levantine", "Egyptian", "Gulf"]


def one_hot(y, dialects):
    return np.array([[1.0 if lab == d else 0.0 for d in dialects] for lab in y])


y_train_oh = one_hot(y_train, dialects)


def fit_defuzz_consequents(mu_train, y_train_oh):
    """Zero-order Sugeno consequent layer: R = argmin_R ||Y - mu_train @ R||^2,
    fit on TRAINING data only. mu_train: (n_train, n_rules). Y: (n_train, n_classes).
    Returns R: (n_rules, n_classes)."""
    R, residuals, rank, sv = np.linalg.lstsq(mu_train, y_train_oh, rcond=None)
    return R


def evaluate(name, mu_train, mu_test, use_defuzz):
    if use_defuzz:
        R = fit_defuzz_consequents(mu_train, y_train_oh)
        crisp_test = mu_test @ R
        pred = np.array(dialects)[crisp_test.argmax(axis=1)]
    else:
        pred = np.array(dialects)[mu_test.argmax(axis=1)]
    acc = accuracy_score(y_test, pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    cm = confusion_matrix(y_test, pred, labels=dialects)
    print("Confusion matrix (rows=true, cols=pred), order:", dialects)
    print(cm)
    print(classification_report(y_test, pred, labels=dialects))
    return acc, pred


# ============================================================
# (a)/(b) ISOTROPIC premise functions
# ============================================================
iso_centroids = {}
iso_variances = {}
for d in dialects:
    mask = y_train == d
    c = z_train[mask].mean(axis=0)
    iso_centroids[d] = c
    sq_dists = np.sum((z_train[mask] - c) ** 2, axis=1)
    iso_variances[d] = sq_dists.mean() / D


def anfis_membership_isotropic(z, centroids, variances, dialects):
    D = z.shape[1]
    log_w = np.array(
        [
            -0.5 * D * np.log(variances[d])
            - np.sum((z - centroids[d]) ** 2, axis=1) / (2 * variances[d])
            for d in dialects
        ]
    )
    log_w -= log_w.max(axis=0, keepdims=True)
    w = np.exp(log_w)
    w_sum = w.sum(axis=0, keepdims=True)
    mu = w / np.clip(w_sum, 1e-12, None)
    return mu.T


mu_iso_train = anfis_membership_isotropic(z_train, iso_centroids, iso_variances, dialects)
mu_iso_test = anfis_membership_isotropic(z_test, iso_centroids, iso_variances, dialects)

acc_a, _ = evaluate("(a) Isotropic premise, NO defuzzification [reproduces v4 = 0.65]",
                     mu_iso_train, mu_iso_test, use_defuzz=False)
acc_b, _ = evaluate("(b) Isotropic premise, WITH defuzzification [NEW]",
                     mu_iso_train, mu_iso_test, use_defuzz=True)

# ============================================================
# (c)/(d) ANISOTROPIC (per-dimension) premise functions
# ============================================================
global_var = z_train.var(axis=0)
VAR_FLOOR_FRACTION = 0.05

aniso_centroids = {}
aniso_variances = {}
for d in dialects:
    mask = y_train == d
    c = z_train[mask].mean(axis=0)
    aniso_centroids[d] = c
    per_dim_var = z_train[mask].var(axis=0)
    per_dim_var = np.maximum(per_dim_var, VAR_FLOOR_FRACTION * global_var)
    aniso_variances[d] = per_dim_var


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


mu_aniso_train = anfis_membership_anisotropic(z_train, aniso_centroids, aniso_variances, dialects)
mu_aniso_test = anfis_membership_anisotropic(z_test, aniso_centroids, aniso_variances, dialects)

acc_c, _ = evaluate("(c) Anisotropic premise, NO defuzzification [reproduces v5]",
                     mu_aniso_train, mu_aniso_test, use_defuzz=False)
acc_d, pred_d = evaluate("(d) Anisotropic premise, WITH defuzzification [NEW]",
                          mu_aniso_train, mu_aniso_test, use_defuzz=True)

# ---- Baseline: k-NN on the same MARBERT+PCA space, for reference ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(z_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(z_test))

print("\n================ SUMMARY ================")
print(f"(a) Isotropic,  no defuzz   : {acc_a:.4f}")
print(f"(b) Isotropic,  + defuzz    : {acc_b:.4f}")
print(f"(c) Anisotropic, no defuzz  : {acc_c:.4f}")
print(f"(d) Anisotropic, + defuzz   : {acc_d:.4f}")
print(f"k-NN (k=5) baseline          : {knn_acc:.4f}")
print("===========================================")

# ---- Qualitative examples: use the best-performing combo (d) for illustration ----
print("\n=== Example fuzzy membership vectors (anisotropic + defuzzified decision) ===")
rng = np.random.RandomState(RANDOM_SEED)
example_pos = rng.choice(len(z_test), 5, replace=False)
for i in example_pos:
    row = df.iloc[test_idx[i]]
    print(
        row["utterance"],
        "-> true:", y_test[i],
        "| predicted:", pred_d[i],
        "| mu (pre-defuzz):", dict(zip(dialects, np.round(mu_aniso_test[i], 3))),
    )
