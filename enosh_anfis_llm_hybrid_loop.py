# ============================================================
# ANFIS-LLM Hybrid Loop — REAL implementation of Appendix B.6
# ------------------------------------------------------------
# Appendix B currently says this architecture is "conceptual
# only; no hybrid system was implemented." This script makes
# that no longer true: every step in the B.6 pseudo-code loop
# is implemented with real, running code -- no placeholder
# functions.
#
#   x_t = f_theta(s_t)     -> real MARBERT sentence embedding
#   z_t = PCA_transform    -> real PCA, fit on TRAIN split only
#   mu_t = ANFIS_forward   -> the REAL trained classifier from
#                             v6 (anisotropic Gaussian premise +
#                             learned defuzzification consequent
#                             layer) -- the same model whose
#                             held-out accuracy you already have
#                             real numbers for. Not re-invented,
#                             just reused as a forward pass.
#   c_t = build_context    -> REAL prompt-steering logic: reads
#                             mu_t and writes a system prompt
#                             that reflects the fuzzy membership
#                             (a hard "respond in X" when one
#                             dialect dominates, a soft "blend of
#                             X and Y" when the input is genuinely
#                             ambiguous between two dialects)
#   y_t_hat = LLM_generate -> REAL call to an LLM API (Gemini by
#                             default; OpenAI left as a commented
#                             alternative)
#
# IMPORTANT: the ANFIS half is fit on the TRAIN split only and
# then run on HELD-OUT TEST examples in the demo loop below, so
# the fuzzy layer you see acting on the LLM is the same evaluated
# model, not something re-fit on data it will then "predict."
#
# ============================================================

!pip install -q transformers google-generativeai

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

# ---- 0. Configure your LLM backend ----
# Get a free Gemini API key at https://aistudio.google.com/apikey
# and paste it below. (Same Google account you used for Opal works.)
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def llm_generate(user_text, context_prompt):
    """Real call to an LLM, conditioned on the ANFIS-derived context."""
    full_prompt = (
        f"{context_prompt}\n\n"
        f"User message (Arabic): {user_text}\n"
        f"Respond naturally and briefly, in Arabic, following the dialect instruction above."
    )
    response = gemini_model.generate_content(full_prompt)
    return response.text.strip()

# --- Alternative backend: OpenAI (uncomment if you'd rather use this) ---
# from openai import OpenAI
# client = OpenAI(api_key="PASTE_YOUR_OPENAI_API_KEY_HERE")
# def llm_generate(user_text, context_prompt):
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": context_prompt},
#             {"role": "user", "content": user_text},
#         ],
#     )
#     return resp.choices[0].message.content.strip()

# ---- 1. Load data ----
DATA_PATH = "Copy of ARABIC_DIALECTS_ANFIS_GOLD (1)(1) (1).xlsx"   # <-- check this matches your Colab Files panel
df = pd.read_excel(DATA_PATH, sheet_name="GOLD_Long_UTT_RESP")

# ---- 2. Load MARBERT (the encoder f_theta from Section B.2) ----
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
    return np.vstack(all_vecs)

print("Embedding utterances with MARBERT (this can take a few minutes)...")
utter_emb = embed_marbert(df["utterance"].astype(str).tolist())
print("MARBERT embedding shape:", utter_emb.shape)

# ---- 3. Train/test split (identical seed to v4/v5/v6, for consistency) ----
labels = df["dialect_name"].values
idx = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
)

# ---- 4. PCA fit on TRAIN ONLY (this is z_t = PCA_transform(x_t) from B.2) ----
pca = PCA(n_components=50, random_state=RANDOM_SEED)
z_train = pca.fit_transform(utter_emb[train_idx])
z_test = pca.transform(utter_emb[test_idx])
y_train = labels[train_idx]
y_test = labels[test_idx]
D = z_train.shape[1]

dialects = ["Levantine", "Egyptian", "Gulf"]

# ---- 5. ANFIS premise layer: anisotropic Gaussian (best-performing from v5/v6) ----
global_var = z_train.var(axis=0)
VAR_FLOOR_FRACTION = 0.05

centroids = {}
variances = {}
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

# ---- 6. Defuzzification consequent layer: R fit on TRAIN ONLY (from v6) ----
def one_hot(y, dialects):
    return np.array([[1.0 if lab == d else 0.0 for d in dialects] for lab in y])

mu_train = anfis_membership(z_train, centroids, variances, dialects)
R, *_ = np.linalg.lstsq(mu_train, one_hot(y_train, dialects), rcond=None)

def anfis_forward(text):
    """Full ANFIS_forward(u_t) from B.6 -- real embedding, real PCA,
    real anisotropic Gaussian premise, real learned defuzzification."""
    emb = embed_marbert([text])
    z = pca.transform(emb)
    mu = anfis_membership(z, centroids, variances, dialects)[0]
    crisp = mu @ R
    dominant_idx = int(np.argmax(crisp))
    return mu, dialects[dominant_idx], crisp

# ---- 7. build_context(mu_t, H_{t-1}) -- prompt-steering strategy from B.5.1 ----
def build_context(mu, dialects, ambiguity_margin=0.15):
    order = np.argsort(mu)[::-1]  # descending
    top1, top2 = dialects[order[0]], dialects[order[1]]
    top1_val, top2_val = mu[order[0]], mu[order[1]]

    if (top1_val - top2_val) < ambiguity_margin:
        # Genuinely fuzzy input: the membership vector doesn't cleanly
        # favor one dialect -- tell the LLM to blend, rather than forcing
        # a hard classification decision it doesn't deserve.
        context = (
            f"The user's message shows mixed dialectal features, primarily between "
            f"{top1} Arabic ({top1_val:.2f} membership) and {top2} Arabic ({top2_val:.2f} membership). "
            f"Respond in a way that leans toward {top1} Arabic but stays natural for a "
            f"speaker who code-switches between {top1} and {top2}."
        )
    else:
        context = (
            f"The user's message is classified as {top1} Arabic "
            f"(membership degree {top1_val:.2f}). Respond fluently in {top1} Arabic, "
            f"matching this dialectal register."
        )
    return context

# ---- 8. Full closed-loop demo on REAL held-out test utterances ----
print("\n=== ANFIS -> LLM closed hybrid loop: live demo on held-out test examples ===\n")
rng = np.random.RandomState(RANDOM_SEED)
example_pos = rng.choice(len(test_idx), 5, replace=False)

for i in example_pos:
    row = df.iloc[test_idx[i]]
    s_t = row["utterance"]
    true_dialect = y_test[i]

    mu_t, predicted_dialect, crisp = anfis_forward(s_t)
    c_t = build_context(mu_t, dialects)
    y_hat = llm_generate(s_t, c_t)

    print("-" * 70)
    print("s_t (input):", s_t)
    print("true dialect:", true_dialect, "| ANFIS predicted:", predicted_dialect)
    print("mu_t:", dict(zip(dialects, np.round(mu_t, 3))))
    print("c_t (context sent to LLM):", c_t)
    print("y_t_hat (LLM response):", y_hat)

print("\nDone. Copy this whole block back to Claude, including any errors.")
