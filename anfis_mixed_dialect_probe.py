# ============================================================
# ANFIS mixed-dialect probe — testing Reviewer 2's concern
# ------------------------------------------------------------
# Reviewer 2's point: the dataset only has hard-labeled, single-
# dialect utterances (LEV, EGY, or GUL), never genuinely mixed
# or code-switched examples with a known true blend. So we can't
# fully prove that the ANFIS model's "fuzzy" membership degrees
# actually track real dialect mixing, rather than just being
# leftover uncertainty from an imperfect classifier.
#
# This script builds a SYNTHETIC test of that idea, using data
# we already have, with NO new data collection needed:
#
#   Your corpus has the SAME 999 scenarios written out in all
#   three dialects (LEV/EGY/GUL versions of the same content).
#   For a sample of scenarios, we build a "mixed" sentence by
#   taking the FIRST HALF of the words from one dialect's version
#   and the SECOND HALF of the words from another dialect's
#   version of the SAME scenario, and gluing them together.
#
#   Example (illustrative): if scenario 17's LEV utterance is
#   "شو صاير معك اليوم" and its EGY utterance is "ايه اللي حصلك
#   النهاردة", a LEV+EGY mix might be "شو صاير حصلك النهاردة".
#
#   We KNOW the true composition of each mixed sentence (half
#   dialect A, half dialect B), because we built it ourselves.
#   That gives us a small amount of genuine ground truth for
#   mixing, which the original dataset does not have.
#
# What we check:
#   1. For PURE (unmixed) held-out test sentences, how confident
#      is the trained model (max membership degree) on average?
#   2. For the SYNTHETIC MIXED sentences, is the model LESS
#      confident on average (as it should be, if it is genuinely
#      sensitive to mixing, not just guessing)?
#   3. For the mixed sentences, does the model's TOP-2 dialects
#      by membership actually match the two dialects that were
#      really mixed in (rather than the unrelated third dialect)?
#
# This does NOT fully answer Reviewer 2 -- it's synthetic,
# word-chopped mixing, not natural code-switching by a real
# bilingual speaker, and nobody manually verified these mixed
# sentences read naturally. We say this honestly in the paper.
# But it is a real, run, quantitative check -- not just a claim.
#
# IMPORTANT: this script re-embeds and re-fits the SAME model as
# v6 (identical anisotropic premise + defuzzification), so the
# model being probed here is exactly the one already reported in
# the paper -- nothing new is trained, we're just testing it on
# new synthetic inputs.
# ============================================================

!pip install -q transformers

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---- 1. Load data ----
DATA_PATH = "Copy of ARABIC_DIALECTS_ANFIS_GOLD (1)(1) (1).xlsx"   # <-- check this matches your Colab Files panel
df = pd.read_excel(DATA_PATH, sheet_name="GOLD_Long_UTT_RESP")

print("\nColumns available in the long-format sheet:")
print(df.columns.tolist())

# ---- 2. Find a scenario-id column so we can match the SAME
#         scenario across its LEV/EGY/GUL versions ----
candidate_names = [
    "scenario_id", "scenario", "id", "ID", "Scenario_ID", "ScenarioID",
    "scenario_num", "scenario_no", "row_id", "index", "Index",
]
scenario_col = None
for c in candidate_names:
    if c in df.columns:
        scenario_col = c
        break

if scenario_col is None:
    # No explicit scenario-id column exists. Since this long-format sheet
    # was built by melting an original wide-format sheet (one row per
    # scenario, one column per dialect), the scenario grouping is still
    # implicit in the ROW ORDER -- we just have to detect the pattern.
    dn = df["dialect_name"].astype(str).tolist()
    n = len(dn)
    print("\nNo scenario-id column found. Detecting scenario grouping from row order instead.")
    print("Unique dialect_name values found:", sorted(set(dn)))
    print("First 12 dialect_name values in row order:", dn[:12])
    print("Last 12 dialect_name values in row order:", dn[-12:])

    def try_interleaved(dn, n):
        # Hypothesis: rows come in groups of 3 -- (i, i+1, i+2) is one
        # scenario written in all 3 dialects.
        if n % 3 != 0:
            return None
        scen_ids = [-1] * n
        for i in range(0, n, 3):
            triple = dn[i:i + 3]
            if len(set(triple)) != 3:
                return None
            for j in range(3):
                scen_ids[i + j] = i // 3
        return scen_ids

    def try_stacked(dn, n):
        # Hypothesis: all of one dialect's rows come first, then all of
        # the next dialect's, then all of the third's -- and position
        # within each block lines up with the same scenario.
        runs = []
        cur, start = dn[0], 0
        for i in range(1, n):
            if dn[i] != cur:
                runs.append((cur, start, i - 1))
                cur, start = dn[i], i
        runs.append((cur, start, n - 1))
        if len(runs) != 3:
            return None
        lengths = [r[2] - r[1] + 1 for r in runs]
        if len(set(lengths)) != 1:
            return None
        scen_ids = [-1] * n
        for _, start, end in runs:
            for pos, i in enumerate(range(start, end + 1)):
                scen_ids[i] = pos
        return scen_ids

    scen_ids = try_interleaved(dn, n)
    pattern_used = "interleaved triples (rows i, i+1, i+2 = one scenario)" if scen_ids else None
    if scen_ids is None:
        scen_ids = try_stacked(dn, n)
        pattern_used = "stacked blocks (each dialect's rows together, aligned by position within block)" if scen_ids else None

    if scen_ids is None:
        print("\n*** Could not detect the scenario ordering automatically either. ***")
        print("Please copy this whole diagnostic block back to Claude:")
        print(" - dialect_name value counts:", df["dialect_name"].value_counts().to_dict())
        raise SystemExit(
            "No usable scenario ordering found -- stopping here, see message above."
        )

    print(f"\nDetected scenario pattern: {pattern_used}")
    df["_scenario_id"] = scen_ids
    scenario_col = "_scenario_id"

print(f"\nUsing '{scenario_col}' as the scenario-id column.")

# Sanity check: each scenario should appear once per dialect (<=3 rows)
counts = df.groupby(scenario_col)["dialect_name"].nunique()
print("Scenarios with all 3 dialects present:", (counts == 3).sum(), "/", len(counts))

# ---- 3. Load MARBERT (identical to v6) ----
model_name = "UBC-NLP/MARBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(DEVICE)
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
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(DEVICE)
        out = bert_model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(pooled.cpu().numpy())
    return np.vstack(all_vecs)


print("\nEmbedding utterances with MARBERT (this can take a few minutes)...")
utter_emb = embed_marbert(df["utterance"].astype(str).tolist())

# ---- 4. Identical train/test split, PCA, and ANFIS model as v6 ----
labels = df["dialect_name"].values
idx = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
)

pca = PCA(n_components=50, random_state=RANDOM_SEED)
z_train = pca.fit_transform(utter_emb[train_idx])
z_test = pca.transform(utter_emb[test_idx])
y_train = labels[train_idx]
y_test = labels[test_idx]

dialects = ["Levantine", "Egyptian", "Gulf"]

global_var = z_train.var(axis=0)
VAR_FLOOR_FRACTION = 0.05
centroids, variances = {}, {}
for d in dialects:
    mask = y_train == d
    c = z_train[mask].mean(axis=0)
    centroids[d] = c
    per_dim_var = z_train[mask].var(axis=0)
    per_dim_var = np.maximum(per_dim_var, VAR_FLOOR_FRACTION * global_var)
    variances[d] = per_dim_var


def anfis_membership(z, centroids, variances, dialects):
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


def anfis_forward_text(text):
    emb = embed_marbert([text])
    z = pca.transform(emb)
    mu = anfis_membership(z, centroids, variances, dialects)[0]
    return mu


# ---- 5. Baseline: confidence on PURE (unmixed) held-out test sentences ----
mu_test_pure = anfis_membership(z_test, centroids, variances, dialects)
pure_confidence = mu_test_pure.max(axis=1)
print("\n=== Baseline: PURE (unmixed) held-out test sentences ===")
print(f"Average top-membership confidence (max mu): {pure_confidence.mean():.4f}")

# ---- 6. Build synthetic mixed-dialect sentences from TEST-split scenarios only ----
test_df = df.iloc[test_idx].copy()
test_df["dialect_name_"] = test_df["dialect_name"].values

# Only use scenarios where at least 2 dialect versions landed in the TEST split
scenario_groups = test_df.groupby(scenario_col)

pairs = [("Levantine", "Egyptian"), ("Levantine", "Gulf"), ("Egyptian", "Gulf")]
mixed_records = []

for scenario_id, group in scenario_groups:
    available = {row["dialect_name"]: row["utterance"] for _, row in group.iterrows()}
    for d_a, d_b in pairs:
        if d_a in available and d_b in available:
            words_a = str(available[d_a]).split()
            words_b = str(available[d_b]).split()
            if len(words_a) < 2 or len(words_b) < 2:
                continue
            mid_a = len(words_a) // 2
            mid_b = len(words_b) // 2
            # A-first-half + B-second-half
            mixed_1 = " ".join(words_a[:mid_a] + words_b[mid_b:])
            # B-first-half + A-second-half
            mixed_2 = " ".join(words_b[:mid_b] + words_a[mid_a:])
            mixed_records.append((scenario_id, d_a, d_b, mixed_1))
            mixed_records.append((scenario_id, d_a, d_b, mixed_2))

print(f"\nBuilt {len(mixed_records)} synthetic mixed-dialect sentences from held-out test scenarios.")

if len(mixed_records) == 0:
    print("No overlapping test-split scenarios found across dialects -- cannot run the mixed probe.")
    print("Copy this message back to Claude along with the scenario-count sanity check printed above.")
    raise SystemExit("No mixed pairs available.")

# ---- 7. Run the trained model on every synthetic mixed sentence ----
results = []
for scenario_id, d_a, d_b, mixed_text in mixed_records:
    mu = anfis_forward_text(mixed_text)
    mu_by_dialect = dict(zip(dialects, mu))
    top2 = sorted(mu_by_dialect, key=mu_by_dialect.get, reverse=True)[:2]
    correct_top2 = set(top2) == {d_a, d_b}
    results.append({
        "scenario_id": scenario_id,
        "dialect_a": d_a,
        "dialect_b": d_b,
        "mixed_text": mixed_text,
        "mu_LEV": mu_by_dialect["Levantine"],
        "mu_EGY": mu_by_dialect["Egyptian"],
        "mu_GUL": mu_by_dialect["Gulf"],
        "max_mu": mu.max(),
        "top2_correct": correct_top2,
    })

results_df = pd.DataFrame(results)

print("\n=== Synthetic MIXED-dialect sentences ===")
print(f"Average top-membership confidence (max mu): {results_df['max_mu'].mean():.4f}")
print(f"(Compare to {pure_confidence.mean():.4f} for pure, unmixed sentences -- "
      f"lower here would mean the model is genuinely less certain on mixed input.)")

pct_correct_top2 = 100 * results_df["top2_correct"].mean()
print(f"\n% of mixed sentences where the model's TOP-2 dialects by membership "
      f"are exactly the 2 dialects that were actually mixed in: {pct_correct_top2:.1f}%")
print("(If this is well above the ~33% you'd expect from random guessing among the "
      "3 possible pairs, the model is picking up on real signal from both source dialects.)")

print("\n=== 8 example mixed sentences with their membership vectors ===")
for _, row in results_df.sample(min(8, len(results_df)), random_state=RANDOM_SEED).iterrows():
    print("-" * 70)
    print(f"Mixed from: {row['dialect_a']} + {row['dialect_b']}")
    print(f"Mixed text: {row['mixed_text']}")
    print(f"mu: LEV={row['mu_LEV']:.3f}  EGY={row['mu_EGY']:.3f}  GUL={row['mu_GUL']:.3f}")
    print(f"Top-2 dialects match the true 2 sources? {row['top2_correct']}")

print("\nDone. Copy this whole block of output back to Claude, including any errors.")
