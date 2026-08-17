"""Shared paths and defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("LEGAL_ASSISTANT_DATA_DIR", PROJECT_ROOT / "data")).resolve()

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
GROUND_TRUTH_PATH = PROCESSED_DIR / "ground_truth.csv"

MAX_CHUNK_CHARS = 2000

DEFAULT_NUM_RESULTS = 5
MAX_PARTS_PER_ARTICLE = 2
OVERFETCH_FACTOR = 4
DEFAULT_BOOSTS = {
    "title": 3.0,
    "chapter": 1.0,
    "section": 1.0,
    "text": 1.0,
}

MODEL = os.getenv("LEGAL_ASSISTANT_MODEL", "openai/gpt-oss-120b")
TOKENS_PER_MINUTE = int(os.getenv("LEGAL_ASSISTANT_TOKENS_PER_MINUTE", "8000"))
MAX_TOKENS = 3000
MAX_SOURCE_CHARS = 2400
REASONING_EFFORT = os.getenv("LEGAL_ASSISTANT_REASONING_EFFORT", "medium")


CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD_TOKENS = 700
MAX_HISTORY_CHARS = 4000
MAX_CONTEXT_CHARS = (
    TOKENS_PER_MINUTE - MAX_TOKENS - PROMPT_OVERHEAD_TOKENS
) * CHARS_PER_TOKEN - MAX_HISTORY_CHARS

TRANSLATION_MODEL = os.getenv("LEGAL_ASSISTANT_TRANSLATION_MODEL", "openai/gpt-oss-20b")
