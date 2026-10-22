# Deployment

**Status:** deployment guide prepared · app locally smoke-tested (Streamlit `AppTest`) ·
**not deployed**.

## Before publishing anything
1. `pytest`, `ruff check .`, `ruff format --check .` pass.
2. `git status` and `git diff --staged` show no data, configs with private paths, model
   files or `runs/` output. `tests/test_cli_and_privacy.py` checks this for tracked files.
3. The app defaults to synthetic data and shows the research-only notice.

## GitHub
Push only reviewed commits. CI (`.github/workflows/ci.yml`) runs lint, tests and the demo.

## Streamlit Community Cloud (optional)
1. share.streamlit.io → *Create app* → pick the repository, branch `main`, main file
   `app/Home.py`.
2. Dependencies come from `requirements.txt` (installs this package with the `app` extra).
3. No secrets are needed. Do not add any.
4. After deploy, check: the app loads, the notice is visible, the synthetic example
   works, an upload with a bad sampling rate shows an error, and the *Model comparison*
   tab shows the "no report" message (`runs/` is not in the repo).
5. Updates redeploy on push. To roll back, revert the commit and push.

**Do not** upload real volunteer recordings to a public deployment. Streamlit holds
uploads in memory, but the host is a third party and logs are outside your control.

## Working with real data
Run everything locally (or on an institution-managed machine): `streamlit run app/Home.py`
after running the pipeline on `data/private/`. Keep the `runs/` folder private until the
numbers have been reviewed for release.
