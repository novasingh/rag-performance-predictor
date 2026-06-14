# Framework Redesign Plan

A standalone, data-driven decision-support tool built on top of the dissertation
*"A Framework for Predicting Retrieval-Augmented Generation System Performance
Based on Relevant Factors."*

## Goal

Turn the research artifacts (RQ1 experiment metrics, the RQ3 predictive model,
and RQ4 validation) into a self-contained web application that lets a user:

1. Predict RAG performance for a proposed knowledge-base configuration
   (domain volatility x dataset freshness x source-type mix) **before deployment**.
2. Receive evidence-based recommendations (update cadence, source mix,
   deployment readiness, risks) that are **computed from the data and the model**,
   not hand-written.
3. Explore how freshness and source diversity affect each metric, and how
   reliable the framework is (RQ4 validation / boundary conditions).

## Three problems being fixed

| Problem | Old behaviour | New behaviour |
|---|---|---|
| Not standalone | `backend/main.py` imported `rq3_experiment.*` and read JSON from `rq1_experiment/`, `rq4_experiment/` via the project root | All code + data bundled under `framework/`. No imports outside `framework/`. |
| Hardcoded numbers | `get_recommendation()` typed `+94%`, `0.164->0.318`, `readiness=85`, `range = readiness/100 * 0.588` | All numbers derived at request time from `condition_metrics.json` + the RQ3 predictor. |
| RQ3 model under-used | only `/predict` used the model; recommendations ignored it | freshness/source curves swept from the trained model; readiness derived from predicted metrics. |

## Architecture

```
framework/
  backend/
    core/config.py       # volatility scores, metric metadata, data paths
    core/predictor.py     # self-contained RAGPerformancePredictor (RQ3 model)
    core/recommender.py   # dynamic recommendations from data + model
    data/                 # bundled artifacts (standalone)
    main.py               # FastAPI app
  frontend/src/App.jsx    # data-driven UI (all numbers from the API)
  sync_data.py            # refresh bundled data from rq1/rq3/rq4
  start.ps1               # standalone launcher
```

### Data layer (`backend/data/`)
Bundled so the app runs with the experiment folders removed:
- `rq3_models.json` — trained OLS/RF/XGBoost + decay curves (the RQ3 model).
- `condition_metrics.json` — per-condition RQ1 metrics (means/std).
- `condition_meta.json` — the 12 conditions (domain, freshness window, source mix).
- `rq4_validation.json` — actual vs predicted + boundary conditions.
- `human_eval_kappa.json` — inter-rater reliability.

`sync_data.py` copies these from the experiment folders so results stay reproducible.

### Prediction (RQ3 model)
`POST /predict` builds the model feature vector
(freshness_score, source_diversity_index, domain_volatility + interactions) from a
user selection and runs the trained OLS coefficients, returning each metric with
its cross-validated MAE band.

### Recommendations (derived, not hardcoded)
`core/recommender.py` computes, per request:
- **Source-impact deltas**: compares the chosen domain's source configurations using
  the real `precision_at_5` means from `condition_metrics.json` (e.g. best vs worst
  source mix), so "+94% / -16%" type findings are calculated, never typed.
- **Update cadence**: derived from the domain's measured freshness sensitivity
  (decay-curve R^2 / freshness coefficient) mapped through transparent thresholds.
- **Deployment readiness**: a weighted score from the model's predicted
  precision, BERTScore and (inverse) hallucination — no fixed numbers.
- **Risks / key findings**: generated from the computed deltas.

### API endpoints
- `GET  /summary` — headline stats (observations, kappa, conditions, boundary counts).
- `POST /predict` — model prediction + derived recommendations for one configuration.
- `GET  /conditions` — all 12 conditions with metadata + measured metrics.
- `GET  /freshness` — model-swept decay curves per domain + best-fit info.
- `GET  /sources` — source-config impact per domain (computed deltas).
- `GET  /validation` — RQ4 actual-vs-predicted + boundary conditions.
- `GET  /model-info` — OLS/RF/XGBoost comparison + feature importances.

### Frontend
`App.jsx` fetches every figure from the endpoints above. No metric value is
written into the component source.

## How to run (standalone)
```powershell
cd framework
./start.ps1
# backend  -> http://localhost:8000  (docs at /docs)
# frontend -> http://localhost:3000
```
