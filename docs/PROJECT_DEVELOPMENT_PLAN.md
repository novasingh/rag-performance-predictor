# Project Development Plan — End to End

**A Framework for Predicting Retrieval-Augmented Generation System Performance Based on Relevant Factors**

Author: Akashdeep Singh · Matric 24072095
Supervisor: AP Dr. Mumtaz Begum Binti Peer Mustafa

This document records the complete development of the project, step by step, in the
order it was actually built: from raw data collection through the four research
phases (RQ1–RQ4) to the deployable predictive framework. It states which sources
were used, how data was scraped and verified, where it is stored, how much was
collected, what each experiment did, and the full technology stack.

---

## 0. Technology Stack & Environment

| Layer | Choice | Notes |
|---|---|---|
| Operating system | **Windows 11** | Development + all experiment runs |
| Language | **Python 3.13** | Single language across data, experiments, backend |
| Environment | `.venv` virtual environment | Dependencies pinned in `requirements.txt` |
| Data ingestion | Custom **staged data-load pipeline** (dlt-style RAW → CLEANED → FINAL) | `requests`, `beautifulsoup4`, `lxml`, `playwright` |
| Web scraping | `requests` + `BeautifulSoup` (static) and **Playwright** (JS-rendered / anti-bot pages) | Headless Chromium fallback |
| Embeddings (local) | **`sentence-transformers/all-MiniLM-L6-v2`** | Runs locally; cached in `.model_cache/` |
| Vector store | **FAISS** (`faiss-cpu`, exact inner-product search) | One index per condition |
| Accuracy scoring | **BERTScore** (backed by all-MiniLM-L6-v2) | Response vs ground-truth similarity |
| Hallucination | NLI entailment check (DeBERTa-style) | Claim-level entailment against retrieved context |
| Statistics / ML | `scikit-learn`, `statsmodels`, `pandas`, `numpy` | ANOVA, OLS, Random Forest, XGBoost, decay curves |
| Backend (framework) | **FastAPI** + Uvicorn | Standalone, stdlib-only predictor |
| Frontend (framework) | **React 19** + Vite 6 + Recharts | Data-driven dashboard |

Core dependencies (`requirements.txt`): `requests`, `pandas`, `sentence-transformers`,
`scikit-learn`, `beautifulsoup4`, `lxml`, `python-dateutil`, `python-dotenv`,
`langdetect`, `playwright`, `bert-score`, `faiss-cpu`, `sentencepiece`.

### Design principle
Two **dataset-level factors** are studied — **dataset freshness** and **source-type
composition** — across three domains chosen for different information volatility:
**Technology (high)**, **Healthcare (medium)**, **History (low)**. The RAG model
architecture (embeddings, retrieval, generation, prompt) is held **constant** across
every condition so that observed differences reflect the dataset factors, not model
choices.

---

## 1. Phase 0 — Data Collection (`dataset_rag_builder/`)

A custom data-load pipeline collects, scrapes, cleans, and stages documents into nine
collections (3 domains × 3 source types). Entry point: `python -m dataset_rag_builder.cli`.

### 1.1 Sources used (by domain × source type)

The collection plan is defined in `dataset_rag_builder/config.py` (`COLLECTIONS`):

| Collection | Domain | Source type | Primary API / source |
|---|---|---|---|
| `technology_academic` | Technology | academic | **arXiv** (cs.IR, cs.AI, cs.LG) + Crossref |
| `technology_news` | Technology | news | **Google News RSS**, **Bing News RSS**, **GDELT** |
| `technology_technical` | Technology | technical | **Official documentation** seed crawl (Kubernetes, MDN, Microsoft Learn, AWS, GCP, Python, React, Docker, PyTorch, TensorFlow, …) |
| `healthcare_academic` | Healthcare | academic | **PubMed / PubMed Central** (E-utilities) |
| `healthcare_news` | Healthcare | news | Google/Bing News RSS, GDELT (health-anchored queries) |
| `healthcare_technical` | Healthcare | technical | **Wikipedia** medical/technical + clinical references |
| `history_academic` | History | academic | **Wikipedia** scholarly history + Crossref |
| `history_news` | History | news | Google/Bing News RSS, GDELT (heritage/archive-anchored) |
| `history_technical` | History | technical | **Archival/heritage seeds** (Library of Congress, National Archives, Europeana, archive.org, IIIF, museums, UNESCO) |

Keyword pools per collection (hundreds of seed phrases) plus the 12 RQ1-aligned
domain queries drive the search/fetch loop (`RQ1_DOMAIN_QUERIES`).

### 1.2 How documents are scraped

Implemented in `functions/sources.py`:
- **API parsers**: `parse_arxiv`, `parse_pubmed`, `parse_crossref`, `parse_wikipedia`,
  `parse_google_news_rss`, `parse_bing_news_rss`, `parse_gdelt`.
- **Article extraction**: `fetch_article_text_requests*` (static HTML via requests +
  BeautifulSoup) with a **Playwright** headless-Chromium fallback
  (`fetch_article_text_playwright*`) for JavaScript-rendered or anti-bot pages.
- **Documentation seed crawls**: curated host allow-lists for technology docs and
  history archives, with child-link discovery and noise/interstitial blocklists.
- **Robustness**: per-host cooldowns, retry rounds, redirect resolution (Google/Bing
  news redirects), tracking-parameter stripping, and CAPTCHA/interstitial detection.

### 1.3 How data is verified / quality-assured

Quality control runs at two points — during cleaning (`functions/processing.py`)
and during dataset assembly (`functions/dataset_builder.py`):

1. **Title/text presence** — drop empty records.
2. **Interstitial / CAPTCHA detection** — `looks_like_access_interstitial` removes
   "verify you are human", Cloudflare challenge, and access-denied pages.
3. **Minimum length** — ≥120 chars for academic/news, ≥200 for technical.
4. **Language filter** — English-only via `langdetect` (`is_english_text`).
5. **Valid publication date** — must parse to a real date (drives freshness).
6. **Domain relevance** — keyword/term-hit thresholds per domain; news has extra
   off-topic guards (sports/finance noise rejection) and domain-anchored checks.
7. **Documentation-type checks** — technology-technical must look like real docs
   (URL/title/text signals); Wikipedia is rejected for that collection.
8. **De-duplication** — `deduplicate_data`: small batches use normalized-text keys;
   larger batches embed with **all-MiniLM-L6-v2** and drop pairs with **cosine ≥ 0.95**.

Each drop reason is counted (`_quality_filter_docs` stats) so attrition is auditable.

### 1.4 Where data is stored (RAW → CLEANED → FINAL)

Output root: `rag_dataset/`. Three staged folders are created per collection
(`ensure_stage_structure`):

```
rag_dataset/
├── raw/        <collection>/    # exactly what was fetched (checkpointed every 25 docs)
├── cleaned/    <collection>/    # after cleaning + relevance + dedup (CSV + JSON)
├── final/      <collection>/    # final records with metadata + freshness fields
│   └── conditions/              # documents bucketed into the 12 RQ1 conditions
├── logs/
└── summary.json                 # collection counts, condition counts, totals
```

Each FINAL document carries: `id`, `title`, `text`, `source_name`, `source_type`,
`domain`, `publication_date`, `published_date`, `freshness_score`, `freshness_label`,
`freshness_window`, `source_api`, `word_count`, `language`, `collected_at`, `url`,
`author`.

**Freshness** is computed per domain volatility (`DOMAIN_VOLATILITY_DAYS`: technology
180d, healthcare 365d, history 3650d) → a continuous score in [0,1] plus a categorical
window.

### 1.5 How much data was collected

From `rag_dataset/summary.json` (generated 2026-05-07, RSS news provider,
RQ1-query-aligned, strict mode):

| Collection | Documents |
|---|---|
| technology_academic | 1,633 |
| technology_news | 1,172 |
| technology_technical | 1,000 |
| healthcare_academic | 1,992 |
| healthcare_news | 1,123 |
| healthcare_technical | 1,000 |
| history_academic | 1,027 |
| history_news | 1,000 |
| history_technical | 1,000 |
| **Total** | **10,787 documents** |

Documents are then bucketed into the **12 experimental conditions** (each comfortably
above the 200-document minimum — `conditions_meet_minimum: true`, zero shortfalls).

---

## 2. Phase 0.5 — Experimental Conditions (the 12-condition design)

Conditions are defined in `dataset_rag_builder/config.py` (`RQ1_CONDITIONS`) and mirrored
in `rq1_experiment/config.py`. They cross freshness windows with source configurations,
domain-adjusted for volatility:

| Condition | Domain | Freshness window | Source configuration |
|---|---|---|---|
| C1 | Technology | ≤ 1 week | Academic only |
| C2 | Technology | ≤ 1 week | Academic + News |
| C3 | Technology | ≤ 1 week | Academic + News + Technical |
| C4 | Technology | 1 wk – 1 mo | Academic only |
| C5 | Technology | 1 wk – 1 mo | Academic + News |
| C6 | Technology | 1 wk – 1 mo | Academic + News + Technical |
| C7 | Healthcare | 1–6 months | Academic only |
| C8 | Healthcare | 1–6 months | Academic + News |
| C9 | Healthcare | 6–12 months | Academic + News + Technical |
| C10 | History | ≥ 6 mo (adj.) | Academic only |
| C11 | History | ≥ 6 mo (adj.) | Academic + Archival |
| C12 | History | ≥ 6 mo (adj.) | Academic + Archival + Reference |

Each condition is evaluated with **200 queries → 2,400 query-response pairs total**.

---

## 3. Phase 1 — RQ1: Factor–Performance Analysis (`rq1_experiment/`)

**Question:** How do dataset freshness and source type relate to RAG performance across
high-, medium-, and low-volatility domains (precision, accuracy, hallucination)?

### 3.1 The RAG system (held constant across all 12 conditions)
`rq1_experiment/rag_system/`:
- `embedder.py` — local **all-MiniLM-L6-v2** sentence embeddings.
- `indexer.py` — builds a **FAISS** index per condition (`results/faiss_indexes/C1…C12`).
- `retriever.py` — dense retrieval, **top-k = 5**, cosine similarity.
- `generator.py` — LLM generation behind a fixed, standardized prompt template
  (`RAG_PROMPT_TEMPLATE`), temperature 0.0 for reproducibility. Backend is configurable
  (local Ollama `llama3.1:8b`, or a hosted generator) but **fixed within a run**.
- `pipeline.py` — orchestrates retrieve → ground → generate for each query.

### 3.2 Query bank
`rq1_experiment/query_bank/query_generator.py` builds a pre-registered bank of
**200 queries per condition** (factual / analytical / comparative; ~20% time-sensitive),
created **before** any results are seen to avoid selection bias. Ground-truth answers
are attached for scoring.

### 3.3 Run
```powershell
python -m rq1_experiment.run_experiment      # produces results/raw_outputs/C*_outputs.json
python -m rq1_experiment.run_analysis        # metrics + ANOVA + regression + RF + plots
```
Auto-resumes via checkpoints every 10 queries.

### 3.4 Metrics computed (`rq1_experiment/evaluation/`)
- **Retrieval** (`retrieval_metrics.py`): Precision@5, nDCG@5.
- **Response accuracy** (`response_metrics.py`): BERTScore F1, ROUGE-L, METEOR.
- **Hallucination** (`hallucination.py`): NLI claim-level entailment against context.
- **Factor metrics** (`factor_metrics.py`): freshness score, **Source Diversity Index**
  (Shannon entropy of the source-type mix).

### 3.5 Statistical analysis (`rq1_experiment/analysis/`)
- `anova.py` — two-way ANOVA (freshness × source type) + Tukey HSD, η² effect sizes.
- `regression.py` — OLS multiple regression + decay-curve fitting.
- `random_forest.py` — 500-tree Random Forest, 10-fold CV, feature importance.
- `visualizer.py` — all plots.

### 3.6 Stored outputs
`rq1_experiment/results/` → `raw_outputs/`, `metrics/` (per-condition + aggregate JSON,
`condition_metrics_matrix.csv`), `analysis/`, `plots/`, `faiss_indexes/`, `human_eval/`.

### 3.7 Headline findings
- **Technology** (high volatility): Full Diversity raises P@5 0.164 → 0.318 (**+94%**).
- **Healthcare** (medium): Full Diversity **reduces** P@5 0.404 → 0.339 (**−16%**) —
  source pollution.
- **History** (low): adding news raises P@5 0.428 → 0.588 (**+37%**).

---

## 4. Phase 2 — RQ2: Source-Type Contribution (`rq2_experiment/`)

**Question:** What is the individual contribution of source-type configurations across
metrics, and does it vary by domain volatility?

RQ2 runs as part of the RQ1 analysis pipeline (it reuses the 2,400 evaluated pairs):
```powershell
python -m rq1_experiment.run_analysis --step anova
python -m rq1_experiment.run_analysis --step plots
```

- Documents are grouped into **Single-Source / Two-Source Mix / Full Diversity** by their
  **Source Diversity Index** (Shannon entropy, treated as a continuous variable).
- **One-way ANOVA** tests whether source level affects each metric; **two-way ANOVA**
  tests the source × domain-volatility interaction; η² reports effect size.
- Results in `rq1_experiment/results/analysis/rq2/` (`source_level_summary.md`,
  `anova_summary.md`, `rq2_anova_results.json`, `rq2_contribution_results.json`).

Key result: source level significantly affects **precision_at_5** (F=22.11, p<0.001) and
**human_eval_score** (F=20.14, p<0.001); its effect is **domain-dependent** (helps
Technology/History, hurts Healthcare).

---

## 5. Phase 3 — RQ3: Predictive Framework (`rq3_experiment/`)

**Question:** Can a framework estimate RAG performance from freshness and source type
across domains?

### 5.1 Build
```powershell
python -m rq3_experiment.run_rq3
```

### 5.2 Features (`rq3_experiment/data.py`, `config.py`)
- **Base**: `freshness_score` (exponential half-life decay, 180-day half-life),
  `source_diversity_index` (Shannon entropy), `domain_volatility` (tech 1.0 / health 0.5 / history 0.0).
- **Interactions**: fresh×diversity, fresh×volatility, source×volatility.
- **Polynomial** (tree models only): freshness², SDI², volatility².

### 5.3 Models (`rq3_experiment/models/`)
- **OLS regression** (`ols_regression.py`) — interpretable equations (6 base+interaction terms).
- **Random Forest** (`random_forest.py`) — 500 trees, grid-searched, 9 features.
- **XGBoost** (`xgboost_model.py`) — gradient-boosted alternative.
- **Decay curves** (`decay_curves.py`) — exponential / polynomial / logistic per domain.
- All evaluated with **10-fold cross-validation** (stratified by domain and source condition).

### 5.4 Predictor API (`rq3_experiment/framework/predictor.py`)
Inputs: `domain`, `avg_age_days`, `source_diversity_index` → outputs each metric's
**expected value with a cross-validated MAE band**. Saved to
`rq3_experiment/results/rq3_models.json`. This trained model is what the deployable
framework serves.

Best CV MAE per metric (Random Forest wins most): bertscore_f1 0.107, precision_at_5 0.216,
ndcg_at_5 0.303, human_eval_score 0.707; hallucination_rate best on OLS (0.391).

---

## 6. Phase 4 — RQ4: Validation & Benchmarking (`rq4_experiment/`)

**Question:** How accurate is the framework on unseen configurations, and what are its
reliable boundaries?

### 6.1 Run
```powershell
python -m rq4_experiment.run_rq4
```

### 6.2 Method (`rq4_experiment/validation/`)
- **Leave-One-Condition-Out (LOCO)** cross-validation: for each of the 12 conditions,
  train on the other 11 and predict the held-out one.
- **1,000-iteration bootstrap** → 95% confidence interval per condition.
- **Boundary-condition detection**: flags conditions whose 95% CI does not intersect the
  y = x diagonal (i.e., where estimates are less reliable).
- Output: `rq4_experiment/results/rq4_validation.json` + 5 LOCO scatter plots.

### 6.3 Findings
- Retrieval-oriented metrics (P@5, nDCG@5, BERTScore F1, hallucination) → **0 boundary
  conditions** (reliable across all domains).
- **Human-eval score** is hardest: 6 boundary conditions (Technology C4–C6, Healthcare C7–C9).
- **History** is the most stable domain (0 boundaries on all metrics).

---

## 7. Human Evaluation (two human reviewers)

To validate the automated metrics, **two independent human reviewers (Rater A and
Rater B)** manually read and scored a **stratified ~20% sample** of responses
(~40 per condition, ~480 total).

- **Scale**: 5-point Likert across four dimensions — **Relevance, Correctness,
  Freshness, Hallucination** — each with a free-text justification note.
- **Stored data** (`rq1_experiment/results/human_eval/`):
  - `rater_A.csv` and `rater_B.csv`
  - Columns: `Question, Answer, Relevance, Correctness, Freshness, Hallucination,
    Relevance_Note, Correctness_Note, Freshness_Note, Hallucination_Note`.
- **Inter-rater agreement**: pooled **Cohen's κ = 0.921** (threshold ≥ 0.70 required
  before scores are accepted into the analysis). Per dimension: relevance 0.978,
  correctness 0.977, freshness 0.965, hallucination 0.764
  (`rq1_experiment/results/metrics/human_eval_kappa.json`).

Only after the two reviewers' agreement cleared the 0.70 threshold were their human
scores merged into the `human_eval_score` used by RQ1–RQ4.

---

## 8. Deployable Predictive Framework (`framework/`)

The trained RQ3 model is packaged into a standalone web app for prospective,
pre-deployment prediction.

- **Backend** (`framework/backend/`, FastAPI): self-contained `core/predictor.py`
  (the RQ3 model), `core/recommender.py` (data-derived recommendations),
  and `backend/data/` (bundled RQ1 metrics, RQ3 model, RQ4 validation, κ, condition
  metadata). `sync_data.py` refreshes these artifacts from the experiments.
- **Frontend** (`framework/frontend/`, React 19 + Vite + Recharts): Research Overview
  (landing), Prediction Tool, Deployment Guide, and evidence dashboards
  (domain, source impact, freshness, validation, model info). Every number is fetched
  from the API — nothing hardcoded.
- **Run**: `run-windows.bat` (Windows), `run-linux.sh` (Linux), `run-macos.sh` (macOS),
  or `start.ps1`. Backend on `:8000` (`/docs`), frontend on `:3000`.

---

## 9. Full Reproduction Order

```powershell
# 0. Environment (Windows 11)
python -m venv .venv ; .\.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium

# 1. Collect + stage data (RAW -> CLEANED -> FINAL, 12 conditions)
python -m dataset_rag_builder.cli --rq1-query-alignment

# 2. RQ1: run RAG over 12 conditions, then metrics + ANOVA + regression + RF
python -m rq1_experiment.run_experiment
python -m rq1_experiment.run_analysis

# 3. (RQ2 is produced by the RQ1 analysis ANOVA/plots steps)

# 4. RQ3: train OLS + Random Forest + XGBoost + decay curves
python -m rq3_experiment.run_rq3

# 5. RQ4: LOCO validation + bootstrap CIs + boundary conditions
python -m rq4_experiment.run_rq4

# 6. Framework app (bundle data + launch)
cd framework ; python sync_data.py ; .\start.ps1
```

---

## 10. Deliverables — Mapping to Research Objectives

| RO / RQ | Deliverable | Where it lives |
|---|---|---|
| RO1 / RQ1 | 9 verified collections; 2,400-query bank; freshness–performance analysis (ANOVA + η²) | `rag_dataset/`, `rq1_experiment/results/` |
| RO2 / RQ2 | Source-type contribution; Source Diversity Index; domain-dependent effects; human-eval κ ≥ 0.70 | `rq1_experiment/results/analysis/rq2/`, `human_eval/` |
| RO3 / RQ3 | Cross-validated predictive framework (OLS/RF/XGBoost) + decay curves with error bounds | `rq3_experiment/results/rq3_models.json` |
| RO4 / RQ4 | LOCO validation, 95% CI benchmark ranges, documented boundary conditions | `rq4_experiment/results/rq4_validation.json` |
| Practical | Standalone prediction app + deployment guidelines | `framework/` |

All five deliverables are complete and reproducible from the steps above.
