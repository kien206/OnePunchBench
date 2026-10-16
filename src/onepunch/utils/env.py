"""Load secrets and paths (HF_TOKEN, HF_HOME, WANDB_API_KEY, ...) from the repo's `.env`.

Called from `onepunch/__init__.py`, so it runs before any onepunch module imports
huggingface_hub (which reads HF_HOME at import time). Variables already set in the
shell win over `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_env(path: str | Path | None = None) -> Path | None:
    """Load `path`, else `$ONEPUNCH_ROOT/.env`, else `<repo>/.env`. Returns the file used, if any."""
    from dotenv import load_dotenv

    candidates = [Path(path)] if path else []
    if os.environ.get("ONEPUNCH_ROOT"):
        candidates.append(Path(os.environ["ONEPUNCH_ROOT"]) / ".env")
    candidates.append(REPO_ROOT / ".env")
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)
            return p
    return None
