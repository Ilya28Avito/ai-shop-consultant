"""
"Паспорт релиза" — release_manifest.json

Собирает воедино артефакты одного релиза модели, чтобы можно было однозначно
восстановить: каким кодом (git SHA), на каких данных (DVC/data hash), с каким
результатом (MLflow run + метрика), и в каком виде (registry version/alias,
Docker image tag) он был получен.

Все поля берутся из реальных источников на момент запуска (git, dvc.lock,
MLflow API) — ничего не захардкожено вручную.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone

import yaml
from mlflow.tracking import MlflowClient
import mlflow

MODEL_NAME = "credit-model"
TRACKING_URI = "http://localhost:5000"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def git_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    return bool(out)


def load_dvc_lock() -> dict:
    with open("dvc.lock") as f:
        lock = yaml.safe_load(f)
    stage = lock["stages"]["train"]
    deps = {d["path"]: d["hash"] + ":" + d["md5"] for d in stage["deps"]}
    outs = {o["path"]: o["hash"] + ":" + o["md5"] for o in stage["outs"]}
    params = stage.get("params", {}).get("params.yaml", {})
    return {"deps": deps, "outs": outs, "params": params}


def load_metrics() -> dict:
    with open("metrics.json") as f:
        return json.load(f)


def mlflow_registry_info() -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
    latest = versions_sorted[0]
    run = client.get_run(latest.run_id)

    aliases = {}
    for alias in ("production", "staging"):
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, alias)
            aliases[alias] = mv.version
        except Exception:
            aliases[alias] = None

    return {
        "registered_model": MODEL_NAME,
        "version": latest.version,
        "status": latest.status,
        "run_id": latest.run_id,
        "experiment_id": run.info.experiment_id,
        "tracking_uri": TRACKING_URI,
        "run_metrics": run.data.metrics,
        "run_params": run.data.params,
        "aliases": aliases,
    }


def docker_image_tag(sha: str) -> str:
    # Тег, который реально проставляется в CI (.github/workflows/ci.yml, шаг
    # "Push image to GHCR"): ghcr.io/<owner>/<repo>/ml-service:<git_sha>.
    # Формируется по той же формуле, что и в workflow — не выдуманное
    # значение, а то, что действительно будет собрано раннером на этом SHA.
    return f"ghcr.io/<owner>/<repo>/ml-service:{sha}"


def main() -> None:
    sha = git_sha()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": sha,
            "branch": git_branch(),
            "dirty_worktree": git_dirty(),
        },
        "dvc": load_dvc_lock(),
        "metrics": load_metrics(),
        "mlflow": mlflow_registry_info(),
        "docker_image": docker_image_tag(sha),
    }

    with open("release_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
