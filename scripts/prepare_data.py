"""Parse every PDF in data/raw into the chunk corpus used by the search index.

    uv run python scripts/prepare_data.py
"""

from legalization_assistant.ingest import build_corpus

if __name__ == "__main__":
    build_corpus()
