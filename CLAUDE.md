# CLAUDE.md

Guidance for Claude Code in this repository.

## Working agreement (learning project)

- Martyna writes the production code in `src/` herself, with help. Do not edit files in `src/` unless she explicitly asks for a specific change. Prefer hints, explanations and review.
- Writing tests in `tests/` is your job. You may create and edit test files there.
- Respond to Martyna in Polish. Code, identifiers, commit messages and code comments are in English.
- Explain briefly what you are doing and why. Work in small steps, one change at a time.
- Never run `git commit`, `git push`, or open or merge a PR without explicit approval in the current message. Never work directly on `main`.
- Before running a command that changes files or git state, say what it does.
- After any change to `src/` or `tests/`, run `python -m pytest` and report the result.

## Commands

All modules use absolute `src.` imports, so run everything from the repo root, using `python -m` for scripts and tests.

```bash
pip install -r requirements-dev.txt          # runtime deps + pytest

python -m src.train                          # initial training: splits raw data, writes model, metrics, training/production CSVs
uvicorn src.api:app --reload                 # serve POST /predict (requires models/model.pkl)
python -m src.retrain                        # drift check on logged predictions; retrains only if drift is found

python -m pytest                             # all tests
python -m pytest tests/test_drift.py::test_drift_detected_when_monthly_charges_shift   # single test
```

No linter or formatter is configured.

## Architecture

Churn-prediction model (Telco Customer Churn dataset) with a serve → log → detect drift → retrain loop. All paths, feature lists and thresholds live in `src/config.py`.

**Data flow**
1. `src/train.py` reads `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (not in git; place it there manually). It holds out 20% as simulated future traffic (`data/simulation/production_traffic.csv`), saves the remaining 80% as `data/training_data.csv`, trains on a train/test split of that, and writes `models/model.pkl` and `reports/metrics.json`.
2. `src/api.py` loads the model at import time and calls `init_db()`. Every `/predict` request is logged to SQLite at `data/predictions.db` (raw input JSON, probability, label) via `src/db.py`. Input validation is the Pydantic `CustomerFeatures` schema in `src/schemas.py`.
3. `src/drift.py` compares logged inputs with `data/training_data.csv` using a two-sample KS test, for `NUMERIC_FEATURES` only. Drift means p < `DRIFT_P_VALUE_THRESHOLD`.
4. `src/retrain.py` runs the drift check. If any feature drifted, it merges training and production CSVs (deduplicated on `customerID`, keeping the last row), refits on everything, and overwrites `models/model.pkl` and `data/training_data.csv`. The drift baseline therefore moves after each retrain. Retraining does not update the metrics.

**Key design points**
- `build_pipeline()` in `src/preprocessing.py` is one sklearn `Pipeline` that starts with `clean_data` as a `FunctionTransformer`. The model accepts raw feature dicts, so the API passes request data through unchanged. Put feature cleaning inside this pipeline, not in callers.
- The schema's `Literal` values, the feature lists in `config.py` and the raw CSV columns must be kept in sync by hand.
- Modules import with `from src.config import X`, so each name is bound when its module loads. Tests isolate state by monkeypatching the name in every module that uses it (e.g. `db.DB_PATH` and `drift.DB_PATH`, see `tests/test_drift.py`). Patching only `config` has no effect.
- `data/`, `models/*.pkl` and `*.db` are gitignored. The only generated artifact committed is `reports/metrics.json`.

## Conventions

- Branches: `feature/MLP-<n>-<short-desc>`, merged into `main` through a PR on GitHub. One feature per branch.
- Constants go in `src/config.py`.
- Stdlib imports first, then third-party, then `src`. Double quotes. SQL keywords in UPPERCASE. Two blank lines between top-level definitions.
- Commit messages in English, lowercase, imperative, e.g. `add drift check with KS test`.