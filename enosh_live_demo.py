# ============================================================
# ENOSH — LIVE demo, backed by the REAL, FINAL trained ANFIS model
# ------------------------------------------------------------
# Updated from the original version to match the final adopted
# model from the paper: MARBERT embeddings + ANISOTROPIC Gaussian
# premise + a learned DEFUZZIFICATION (consequent) layer, instead
# of the original MiniLM + isotropic-only version (46.33% -> now
# 85.33% held-out accuracy). This script is self-contained: it
# re-embeds, refits PCA, and refits the ANFIS layer from scratch,
# so it works even after a fresh Colab runtime restart.
#
# NEW in this version:
#   - Anisotropic Gaussian premise + defuzzification consequent
#     layer (same model as v6 / the paper's final result)
#   - A small floating badge in the corner of the page that shows
#     the detected dialect at a glance, separate from the bigger
#     results panel with the mu bars
#   - Footer accuracy updated to the real final number (85.33%)
#
# HOW TO USE:
#   1. Run this cell as a NEW CELL in your Colab session (it does
#      not depend on any earlier cell -- everything it needs is
#      loaded/re-fit inside this script).
#   2. Wait for MARBERT to load/download and the utterances to be
#      re-embedded (a few minutes on CPU), then ~15-20 seconds for
#      the tunnel to start.
#   3. It will print a line like:
#        YOUR LIVE DEMO IS READY: https://xxxx-xxxx.trycloudflare.com
#   4. Open that link (on the venue's screen, or as a QR code) --
#      that IS the demo. Anyone who opens it can type any Arabic
#      sentence and see REAL, live-computed mu scores from the
#      final trained model.
#   5. KEEP THIS CELL RUNNING for the whole demo. Stopping it, or
#      Colab disconnecting, takes the public link down. If that
#      happens, just re-run the cell -- you'll get a NEW link, so
#      re-open/re-scan it.
#
# A rehearsal tip: run this once before your talk and actually
# open the link on your phone to confirm it works on the venue's
# wifi, since conference wifi can be unpredictable.
# ============================================================

!pip install -q flask transformers
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, Response
import threading, subprocess, time, re

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---- 1. Load data ----
DATA_PATH = "ARABIC_DIALECTS_ANFIS_GOLD.xlsx"   # <-- check this matches your Colab Files panel
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
    return np.vstack(all_vecs)

print("Embedding utterances with MARBERT (this can take a few minutes)...")
utter_emb = embed_marbert(df["utterance"].astype(str).tolist())
print("MARBERT embedding shape:", utter_emb.shape)

# ---- 3. Train/test split (identical seed to the paper's final model) ----
labels = df["dialect_name"].values
idx = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
)

# ---- 4. PCA fit on TRAIN ONLY ----
pca = PCA(n_components=50, random_state=RANDOM_SEED)
z_train = pca.fit_transform(utter_emb[train_idx])
y_train = labels[train_idx]
D = z_train.shape[1]

dialects = ["Levantine", "Egyptian", "Gulf"]

# ---- 5. Anisotropic Gaussian premise (the adopted model) ----
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

# ---- 6. Defuzzification consequent layer: R fit on TRAIN ONLY ----
def one_hot(y, dialects):
    return np.array([[1.0 if lab == d else 0.0 for d in dialects] for lab in y])

mu_train = anfis_membership(z_train, centroids, variances, dialects)
R, *_ = np.linalg.lstsq(mu_train, one_hot(y_train, dialects), rcond=None)

print("Model ready: anisotropic premise + defuzzification, MARBERT embeddings.")

# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)

PAGE_HTML = """<!doctype html>
<html lang="ar">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ENOSH — Live Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b1120; --panel:#131c2e; --panel-border:#233149;
    --text:#e8edf7; --muted:#8fa0bd;
    --lev:#60a5fa; --egy:#fbbf24; --gul:#34d399;
    --accent:#a78bfa;
  }
  *{box-sizing:border-box}
  body{
    margin:0; min-height:100vh; background:
      radial-gradient(circle at 20% -10%, #1b2b4d 0%, transparent 55%),
      radial-gradient(circle at 90% 10%, #2a1f4d 0%, transparent 45%),
      var(--bg);
    color:var(--text); font-family:'Inter',sans-serif;
    display:flex; flex-direction:column; align-items:center;
    padding:32px 16px 60px;
  }
  .brand{display:flex; align-items:center; gap:10px; margin-bottom:6px}
  .brand .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 14px var(--accent)}
  h1{font-family:'Tajawal',sans-serif; font-weight:900; font-size:clamp(28px,5vw,44px); margin:0; letter-spacing:.5px}
  .subtitle{color:var(--muted); font-size:15px; max-width:560px; text-align:center; margin:10px 0 36px; line-height:1.6}
  .card{
    width:100%; max-width:640px; background:var(--panel);
    border:1px solid var(--panel-border); border-radius:20px;
    padding:24px; box-shadow:0 20px 60px rgba(0,0,0,.35);
  }
  .input-row{display:flex; gap:10px}
  textarea{
    flex:1; min-height:64px; resize:vertical;
    background:#0d1526; border:1px solid var(--panel-border); border-radius:12px;
    color:var(--text); font-family:'Tajawal',sans-serif; font-size:19px;
    direction:rtl; text-align:right; padding:14px 16px; line-height:1.6;
  }
  textarea:focus{outline:2px solid var(--accent); outline-offset:1px}
  button{
    background:var(--accent); color:#150f2b; border:none; border-radius:12px;
    padding:0 22px; font-weight:700; font-size:15px; cursor:pointer;
    transition:transform .1s ease, opacity .15s ease;
  }
  button:hover{opacity:.92} button:active{transform:scale(.97)}
  button:disabled{opacity:.5; cursor:default}
  .hint{color:var(--muted); font-size:12.5px; margin-top:10px; direction:rtl; text-align:right}
  .results{margin-top:26px; display:none}
  .results.show{display:block}
  .analyzed-text{
    direction:rtl; text-align:right; font-family:'Tajawal',sans-serif;
    font-size:20px; padding:14px 16px; background:#0d1526;
    border-radius:12px; border:1px solid var(--panel-border); margin-bottom:20px;
  }
  .bar-row{display:flex; align-items:center; gap:12px; margin-bottom:14px}
  .bar-label{width:92px; font-size:13.5px; color:var(--muted); font-weight:600; letter-spacing:.3px}
  .bar-track{flex:1; height:14px; background:#0d1526; border-radius:8px; overflow:hidden; border:1px solid var(--panel-border)}
  .bar-fill{height:100%; border-radius:8px; transition:width .5s cubic-bezier(.22,1,.36,1)}
  .bar-val{width:52px; text-align:left; font-variant-numeric:tabular-nums; font-size:13.5px; color:var(--text); font-weight:600}
  .dominant{
    margin-top:18px; padding:14px 16px; border-radius:12px;
    background:rgba(167,139,250,.12); border:1px solid rgba(167,139,250,.35);
    display:flex; justify-content:space-between; align-items:center;
  }
  .dominant .label{color:var(--muted); font-size:13px}
  .dominant .value{font-size:19px; font-weight:700; color:var(--accent)}
  .sumcheck{margin-top:14px; font-size:12px; color:var(--muted); text-align:center; font-variant-numeric:tabular-nums}
  .sumcheck b{color:#34d399}
  .loading{color:var(--muted); font-size:14px; margin-top:20px; display:none}
  .loading.show{display:block}
  .error{color:#fb7185; font-size:14px; margin-top:16px; display:none}
  .error.show{display:block}
  .footer{color:var(--muted); font-size:12px; margin-top:28px; text-align:center; max-width:520px; line-height:1.6}
  @media (max-width:480px){ .bar-label{width:70px; font-size:12px} }

  /* --- small floating language-indicator badge --- */
  .lang-badge{
    position: fixed; top: 18px; right: 18px;
    display:flex; align-items:center; gap:8px;
    background: var(--panel); border:1px solid var(--panel-border);
    border-radius: 999px; padding: 9px 16px 9px 14px;
    font-size: 13px; font-weight:700; color: var(--text);
    box-shadow: 0 10px 30px rgba(0,0,0,.4);
    opacity: 0; transform: translateY(-10px) scale(.96);
    transition: opacity .35s ease, transform .35s ease, background .35s ease;
    z-index: 1000; pointer-events:none;
  }
  .lang-badge.show{ opacity:1; transform: translateY(0) scale(1); }
  .lang-badge .lang-dot{ width:9px; height:9px; border-radius:50%; flex:none; transition: background .35s ease; box-shadow: 0 0 10px currentColor; }
  .lang-badge .lang-text{ letter-spacing:.2px; white-space:nowrap; }
  @media (max-width:480px){
    .lang-badge{ top:10px; right:10px; padding:7px 12px 7px 10px; font-size:11.5px; }
  }
</style>
</head>
<body>
  <div class="lang-badge" id="langBadge">
    <span class="lang-dot" id="langBadgeDot"></span>
    <span class="lang-text" id="langBadgeText">—</span>
  </div>

  <div class="brand"><span class="dot"></span><span style="color:var(--muted); font-size:13px; letter-spacing:1px; text-transform:uppercase">Live model demo</span></div>
  <h1>ENOSH</h1>
  <p class="subtitle">Type any Arabic sentence below. This calls the real, trained ANFIS layer from the paper — the same model evaluated on the held-out test set — and returns its actual fuzzy membership scores.</p>

  <div class="card">
    <div class="input-row">
      <textarea id="txt" placeholder="اكتب جملة بأي لهجة عربية..."></textarea>
    </div>
    <div style="display:flex; justify-content:flex-end; margin-top:10px">
      <button id="send">Analyze</button>
    </div>
    <div class="hint">اضغط Enter مع Shift لسطر جديد، أو Enter وحدها للإرسال</div>

    <div class="loading" id="loading">Running the real model…</div>
    <div class="error" id="error"></div>

    <div class="results" id="results">
      <div class="analyzed-text" id="analyzedText"></div>

      <div class="bar-row">
        <div class="bar-label">Levantine</div>
        <div class="bar-track"><div class="bar-fill" id="bar-Levantine" style="background:var(--lev); width:0%"></div></div>
        <div class="bar-val" id="val-Levantine">0.00</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Egyptian</div>
        <div class="bar-track"><div class="bar-fill" id="bar-Egyptian" style="background:var(--egy); width:0%"></div></div>
        <div class="bar-val" id="val-Egyptian">0.00</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Gulf</div>
        <div class="bar-track"><div class="bar-fill" id="bar-Gulf" style="background:var(--gul); width:0%"></div></div>
        <div class="bar-val" id="val-Gulf">0.00</div>
      </div>

      <div class="dominant">
        <span class="label">Dominant dialect (after defuzzification)</span>
        <span class="value" id="dominantVal">—</span>
      </div>
      <div class="sumcheck">Σμ = <b id="sumVal">—</b> (real fuzzy inference — normalized firing strengths, not a canned number)</div>
    </div>
  </div>

  <div class="footer">Powered by the zero-order Sugeno ANFIS layer trained and evaluated in "Integrating Adaptive Neuro-Fuzzy Inference Systems and Large Language Models for Arabic Dialect Processing" — anisotropic Gaussian premise functions with a learned defuzzification layer, fit on an 80/20 held-out split, 85.33% test accuracy.</div>

<script>
const txt = document.getElementById('txt');
const sendBtn = document.getElementById('send');
const loading = document.getElementById('loading');
const errorBox = document.getElementById('error');
const results = document.getElementById('results');
const langBadge = document.getElementById('langBadge');
const langBadgeDot = document.getElementById('langBadgeDot');
const langBadgeText = document.getElementById('langBadgeText');

const dialectColor = {Levantine:'var(--lev)', Egyptian:'var(--egy)', Gulf:'var(--gul)'};
const dialectLabel = {Levantine:'Levantine Arabic', Egyptian:'Egyptian Arabic', Gulf:'Gulf Arabic'};

async function analyze(){
  const text = txt.value.trim();
  if(!text) return;
  errorBox.classList.remove('show');
  results.classList.remove('show');
  loading.classList.add('show');
  sendBtn.disabled = true;

  try{
    const res = await fetch('/analyze', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || 'Request failed');

    document.getElementById('analyzedText').textContent = data.text;
    for(const d of ['Levantine','Egyptian','Gulf']){
      const v = data.mu[d] ?? 0;
      document.getElementById('bar-'+d).style.width = (v*100).toFixed(1)+'%';
      document.getElementById('val-'+d).textContent = v.toFixed(3);
    }
    document.getElementById('dominantVal').textContent = data.dominant_dialect;
    document.getElementById('sumVal').textContent = data.sum_check.toFixed(4);
    results.classList.add('show');

    // update the small floating language-indicator badge
    langBadgeDot.style.background = dialectColor[data.dominant_dialect] || 'var(--accent)';
    langBadgeText.textContent = dialectLabel[data.dominant_dialect] || data.dominant_dialect;
    langBadge.classList.add('show');
  }catch(err){
    errorBox.textContent = 'Could not reach the model: ' + err.message;
    errorBox.classList.add('show');
  }finally{
    loading.classList.remove('show');
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', analyze);
txt.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    analyze();
  }
});
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return Response(PAGE_HTML, mimetype="text/html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    emb = embed_marbert([text])
    z = pca.transform(emb)
    mu = anfis_membership(z, centroids, variances, dialects)[0]
    mu_dict = {d: float(round(float(m), 4)) for d, m in zip(dialects, mu)}

    # decision uses the DEFUZZIFIED crisp score (mu @ R), matching the
    # final adopted model -- not a raw argmax over mu.
    crisp = mu @ R
    dominant = dialects[int(np.argmax(crisp))]

    return jsonify({
        "text": text,
        "mu": mu_dict,               # raw membership degrees, for the bars
        "dominant_dialect": dominant,  # defuzzified decision, for the badge + label
        "sum_check": round(sum(mu_dict.values()), 4),  # should print ~1.0
    })

def run_flask():
    app.run(host="0.0.0.0", port=5000)

threading.Thread(target=run_flask, daemon=True).start()
time.sleep(2)

proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:5000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

print("Starting tunnel, this can take 10-20 seconds...")
for line in proc.stdout:
    m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
    if m:
        print("\n" + "=" * 60)
        print("YOUR LIVE DEMO IS READY:")
        print(m.group(0))
        print("Open this link (or turn it into a QR code) to use the demo.")
        print("Keep this cell RUNNING for the whole demo.")
        print("=" * 60 + "\n")
        break
