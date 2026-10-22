# Contributing

Research prototype for PCOS screening from ECG. See `knowledge.md` for decisions and
status, `docs/architecture.md` for the module map.

## Commands
- `pytest` · `ruff check .` · `ruff format --check .`
- `pcos-ecg demo` (synthetic end to end) · `streamlit run app/Home.py`

## Rules
- Never commit or print participant data: no rows, IDs, waveforms or models from
  `data/private/`. Reports hold counts and metrics only.
- Never invent results, sample sizes, consent/ethics facts or paper numbers. Unknowns stay
  "pending" / "not yet documented".
- Anything that learns (imputer, scaler, selector, SMOTE, model) goes inside the
  per-fold `Pipeline`. Split by participant. No threshold or hyperparameter tuning on
  evaluation folds.
- Failed measurements are NaN plus a reason, never 0.
- Study-data commands must not fall back to demo data.
- Style: small functions, type hints, ruff (line length 100), comments only where the
  reason is not obvious.
