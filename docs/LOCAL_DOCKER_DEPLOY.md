# Running the notebooks + the v2 Docker image locally

Guide for TAs: how to get from a fresh clone of this branch to a running
model-serving container on `http://localhost:5001`, using the two notebooks
(`notebooks/01_training.ipynb`, `notebooks/02_deployment.ipynb`) and the v2
serving image (`docker/Dockerfile.model-serve`).

## What is on this branch (and why)

The `v2` branch ships the serving image but the old monolithic
`notebooks/main.ipynb`, and it gitignores `models/model/` and `artifacts/` -
so a fresh snapshot of `v2` could not even build the image (`COPY` fails).
This branch fixes that:

| Change | Why it was necessary |
|---|---|
| `notebooks/01_training.ipynb` + `02_deployment.ipynb` added (adapted from `main`) | The student-facing train/deploy flow did not exist on v2 |
| Notebooks import `SimpleCNN` / `SimpleLoggingWrapper` from `src.*` instead of defining them inline | The exported pyfunc model is **pickled with those classes**. The Docker image can only unpickle classes from importable modules - a class defined in a notebook (`__main__`) makes `mlflow models serve` fail inside the container with `Can't get attribute 'SimpleLoggingWrapper' on <module '__main__'>` |
| Training notebook downloads data via `src.data.fetch` (no `dvc pull`) | v2 does not track `data/` with DVC at the repo root (`data/` is gitignored); `src.data.fetch.pull_data_from_dvc` downloads the metadata CSVs and all listed images straight from the Hetzner object store |
| `models/model/` and `artifacts/drift_baseline_stats.json` are committed (`.gitignore` adjusted) | `Dockerfile.model-serve` does `COPY models/model ...` and `COPY artifacts/drift_baseline_stats.json ...` - on a fresh v2 clone those paths do not exist and the build fails. Committing them makes the image buildable from any fresh clone. (In CI they are produced by the `train` job and passed as workflow artifacts instead - see `.github/workflows/train.yml`) |
| `pyproject.toml` gained `[build-system]` + explicit `[tool.setuptools] packages = ["src", ...]` | Without it, setuptools' src-layout auto-discovery installs the *contents* of `src/` (`config`, `deployment`, ...) as top-level modules, so `pip install .` did **not** provide `src.deployment.logging_wrapper` (the pickle reference). The container only worked by accident, because `python -m src.deployment.serve` runs with `/app` on `sys.path` next to the `COPY src/ ./src/`. With the explicit packages the installed package really is `src` |
| `notebooks/requirements.txt` updated | Notebooks now need the project deps + Jupyter; `dvc` removed (not needed at the repo root anymore) |
| `scripts/deploy_local.sh` added (notebooks left as close to main as possible) | Builds `docker/Dockerfile.model-serve`, runs it on `:5001`, waits for `/health` and smoke-tests `/health`, `/invocations`, `/infer`, `/drift` - the local equivalent of the production hand-off |
| Promotion cell in `02` guarded for reruns | When the best run does not beat the current Production model, the original cell crashed on `runs:/None/...`; now it keeps Production and continues |

## Prerequisitesv

- Docker Desktop running (`docker version` works)
- `uv` and Python 3.13 (`.python-version`)
- Network access to `https://itudsip.hel1.your-objectstorage.com` (only needed
  for the **full training run** in `01_training.ipynb`; the quick start below
  works without it, using the committed model)

## Setup (once per clone)

```bash
cd <repo root>
uv venv                                  # Python 3.13
source .venv/bin/activate
uv pip install -e .                      # installs the src package + all deps
uv pip install -r notebooks/requirements.txt
python -m ipykernel install --user --name itu-sdse
jupyter lab notebooks/
```

The `-e .` install matters: the notebooks do `from src.models.train import ...`
and the pickled model references `src.deployment.logging_wrapper`.

## Quick start - serve the committed model in Docker (no training)

Works on a fresh clone, no data download, no MLflow state:

```bash
docker build -t vial-model-serve:local -f docker/Dockerfile.model-serve .
docker run -d --name vial-model-serve-local -p 5001:5001 vial-model-serve:local
# wait for torch to load (~30-60s): poll until 200
curl -s http://localhost:5001/health        # -> {"status": "ok"}
```

Send an image (base64 is the portable contract):

```bash
B64=$(base64 -i validation/fixtures/Glass_039_TLC00-20240327111626-061101.jpg | tr -d '\n')
curl -s http://localhost:5001/invocations -H 'Content-Type: application/json' \
    -d "{\"instances\": [\"$B64\"]}"
# -> {"predictions": {"predictions": [1], "confidences": [...], ...}}
curl -s http://localhost:5001/infer -H 'Content-Type: application/json' -d "{\"instances\": [\"$B64\"]}"   # alias
curl -s http://localhost:5001/drift
```

Endpoints exposed by the image (see `infra/digitalocean/app.yaml` for the
production ingress):

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | `{"status": "ok"}` when loaded; 503 `{"status": "loading"}` while torch imports |
| `/invocations` | POST | MLflow pyfunc contract: `{"instances": [<base64 image>]}` |
| `/infer` | POST | alias of `/invocations` |
| `/drift` | GET | KS drift check over the container's own `inference_log.json` vs the baked-in baseline (`?days=N`) |

Stop/remove: `docker rm -f vial-model-serve-local`

## Full flow - train, then deploy (what the students see)

Run the notebooks top to bottom from `jupyter lab` started at the repo root:

1. **`01_training.ipynb`** - downloads metadata + images from the object store
   (`src.data.fetch`), trains the CNN (5 epochs, CPU is fine), writes
   `artifacts/` (checkpoint, results, drift baseline) and `mlflow-state/`
   (SQLite registry), logs the pyfunc model and registers `cnn_model` in
   Staging with the `best` alias.
2. **`02_deployment.ipynb`** - picks the best run, promotes it to Production if
   it beats the current one, exports the pyfunc model to `models/model/`
   (this is what gets baked into the image), serves it inline on `:5001`
   (FastAPI wrapper + `mlflow models serve` subprocess) and tests
   `/health`, `/version`, `/infer`, `/drift`.

After running the notebooks, rebuild the image so it contains the *newly*
exported model, run it, and smoke-test the endpoints - all in one step:

```bash
./scripts/deploy_local.sh          # build + run on :5001 + smoke-test
./scripts/deploy_local.sh stop     # stop/remove the container
```

Manual equivalent:

```bash
docker build -t vial-model-serve:local -f docker/Dockerfile.model-serve .
docker rm -f vial-model-serve-local || true
docker run -d --name vial-model-serve-local -p 5001:5001 vial-model-serve:local
```

The compose alternative (trains inside Docker, state in a named volume):

```bash
docker compose -f docker/docker-compose.yml run --rm train      # needs object-store access
docker compose -f docker/docker-compose.yml build model-serve
docker compose -f docker/docker-compose.yml up -d
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker build` fails: `COPY models/model: not found` (or `artifacts/drift_baseline_stats.json`) | Fresh clone of a branch where they are gitignored. On this branch they are committed; on plain `v2` run `01_training.ipynb` + `02_deployment.ipynb` first (or `docker compose ... run --rm train`) to produce them |
| Container starts but `/health` stays 503 `loading`, then `error`; logs show `Can't get attribute 'SimpleLoggingWrapper' ... __main__` | The model was pickled from a notebook that defined the wrapper/CNN inline. Re-export using the notebooks on this branch (they import from `src.*`) and rebuild the image |
| `docker run` fails: port 5001 already in use | The notebook's inline server or a previous container is still up. Run the notebook's cleanup cell, or `docker rm -f vial-model-serve-local`, or `pkill -f "mlflow models serve"` |
| `pip install -e .` then `import src` fails | Old-style install without the explicit `packages` in `pyproject.toml`; on this branch `uv pip install -e .` from the repo root provides `src`. Running from the repo root also works without install (repo root is on `sys.path`) |
| `01_training.ipynb` fails at the download cell | No access to the object store, or `data/` metadata missing. v2 fetches metadata itself; on `main` the metadata CSVs are DVC-tracked and `dvc pull` is required first |
| `/drift` returns "no logs" | The container logs to its own `/app/artifacts/inference_log.json`; send a few `/invocations` requests first. The baseline must be baked in (`artifacts/drift_baseline_stats.json` at build time) |
| `.dockerignore` "hides" files from the image | The build context excludes `notebooks/`, `docs/`, `data/`, `platform/`, `.git`, `.venv`, `*.db` - only `pyproject.toml`, `README.md`, `src/`, `models/model`, `artifacts/drift_baseline_stats.json` reach the image |

## How CI does it differently

`.github/workflows/train.yml` runs `python -m src.pipeline` +
`python -m src.deployment.export`, uploads `mlflow-state/`, `models/model` and
the drift baseline as workflow artifacts, and the `build` job downloads them
into a fresh checkout before `docker build` - so the model never enters git
there. This branch instead commits one exported model so that a plain
snapshot of the repo is deployable out of the box for local testing.
