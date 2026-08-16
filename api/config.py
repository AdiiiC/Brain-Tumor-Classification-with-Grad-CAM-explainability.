"""Runtime configuration and build metadata."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

API_VERSION = "3.0.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "efficientnetb1-v1")

# README documents T=1.5 from post-training calibration; keep that as the default
# so deployed confidences are calibrated unless deliberately overridden.
DEFAULT_CALIBRATION_TEMP = 1.5


def calibration_temp() -> float:
    return float(os.getenv("CALIBRATION_TEMP", str(DEFAULT_CALIBRATION_TEMP)))


def allowed_origins() -> list[str]:
    """
    CORS origins from ALLOWED_ORIGINS (comma-separated).

    Defaults to local dev hosts. A wildcard is refused when credentials are
    enabled because browsers reject that combination anyway.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if not origins:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return origins


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Short commit SHA, for tracing a prediction back to the code that made it."""
    env_sha = os.getenv("GIT_SHA") or os.getenv("RENDER_GIT_COMMIT")
    if env_sha:
        return env_sha[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def build_info() -> dict[str, str]:
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "git_sha": git_sha(),
    }
