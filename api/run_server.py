from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = REPO_ROOT / ".venv310" / "Lib" / "site-packages"

if SITE_PACKAGES.exists():
    sys.path.insert(0, str(SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.figure_pipeline_api:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
