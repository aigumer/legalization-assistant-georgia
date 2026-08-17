"""Parse every PDF in data/raw into the chunk corpus used by the search index.

    uv run python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legalization_assistant.ingest import build_corpus  # noqa: E402

if __name__ == "__main__":
    build_corpus()
