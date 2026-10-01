# Integrating Adaptive Neuro-Fuzzy Inference Systems and Large Language Models for Arabic Dialect Processing

Supplementary material for the paper submitted to **ISMSIT 2026** (Paper ID 296).

**Authors:** Ibrahim Youssef El-Noshokaty, Shaimaa EL-Ragae, Rahma Mohamed Mohamed — Enosh Science Research Center, Cairo, Egypt.

## What this project is

This project studies how Arabic dialects (Levantine, Egyptian, and Gulf) can be told apart and handled by AI language models in a graded, "fuzzy" way rather than forced into one rigid label. It combines sentence embeddings, a trained Adaptive Neuro-Fuzzy Inference System (ANFIS) classifier, and a working demo (**ENOSH**) that uses the model's confidence scores to steer an LLM's responses toward the detected dialect.

The final trained model reaches **85.33% held-out accuracy** across the three dialects (confirmed with 5-fold cross-validation, 84.99% mean accuracy), well above a nearest-neighbor baseline.

## What's in this repository

- **`paper/`** — the submitted paper (`.tex` and compiled `.pdf`) and the full Response to Reviewers letter.
- **`figures/`** — the t-SNE visualization and the ENOSH demo screenshot used in the paper.
- **`code/`** — standalone Python scripts (written for Google Colab, but runnable anywhere with the listed packages):
  - `anfis_final_model_marbert_anisotropic_defuzz.py` — trains and evaluates the final ANFIS classifier reported in the paper (MARBERT embeddings, anisotropic Gaussian premise, learned defuzzification layer).
  - `anfis_statistical_robustness.py` — the statistical checks reported in the paper: a silhouette score and permutation test on the embedding space, plus 5-fold cross-validation of the classifier.
  - `anfis_mixed_dialect_probe.py` — the synthetic mixed-dialect probe used to partially test whether the model's fuzzy membership scores respond sensibly to dialect mixing.
  - `enosh_anfis_llm_hybrid_loop.py` — the real, running closed-loop pipeline: trained ANFIS membership scores converted into a dialect-aware steering prompt for an LLM decoder.
  - `enosh_live_demo.py` — the ENOSH interactive demo app shown in the paper's figure.
- **`notebooks/full_pipeline_with_results.ipynb`** — a single notebook with all of the above run end-to-end, including the actual output and numbers used in the paper.

## Requirements

The scripts were run in Google Colab and rely on `transformers`, `sentence-transformers`, `scikit-learn`, `pandas`, and `torch`. Each script installs what it needs at the top via `pip install`.

## Data

The experiments use a gold-standard Arabic dialect dataset (999 dialogue scenarios, each in three parallel dialects) created by the authors. It is not included in this repository; contact the authors for access.

## Citation

If you use this work, please cite the paper once published (ISMSIT 2026, Paper ID 296). Citation details will be added here after publication.
