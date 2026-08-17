"""Shared paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pick up ANTHROPIC_API_KEY (and any overrides below) from a local .env file.
load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
GROUND_TRUTH_PATH = PROCESSED_DIR / "ground_truth.csv"

# Articles longer than this are split into parts on paragraph boundaries, so a
# single retrieved chunk stays focused enough to be a useful search unit.
MAX_CHUNK_CHARS = 2000

# Retrieval defaults. Article titles carry the topical words a user actually
# types ("residence permit", "visa"), so they are boosted over body text. These
# weights come from scripts/evaluate_retrieval.py --sweep: boosting `chapter` as
# well hurt, because chapter headings match every article beneath them.
DEFAULT_NUM_RESULTS = 5
# At most this many parts of the same article may occupy result slots, so a long
# article cannot crowd out every other article bearing on the question.
MAX_PARTS_PER_ARTICLE = 2
OVERFETCH_FACTOR = 4
DEFAULT_BOOSTS = {
    "title": 3.0,
    "chapter": 1.0,
    "section": 1.0,
    "text": 1.0,
}

MODEL = os.getenv("LEGAL_ASSISTANT_MODEL", "openai/gpt-oss-120b")
# gpt-oss reasons before answering, and reasoning tokens share this budget with
# the answer text. Groq counts `max_completion_tokens` in full against the
# tokens-per-minute quota (8,000/min on the free tier) whether or not they are
# used, so this is deliberately modest: prompt + budget must fit under it.
MAX_TOKENS = 3000
# Cap on any single excerpt, so one very long article cannot blow the budget.
MAX_SOURCE_CHARS = 2400
REASONING_EFFORT = os.getenv("LEGAL_ASSISTANT_REASONING_EFFORT", "medium")

# The corpus is English, so a query written in another script cannot match
# anything. A smaller, faster model turns such queries into English search terms.
TRANSLATION_MODEL = os.getenv("LEGAL_ASSISTANT_TRANSLATION_MODEL", "openai/gpt-oss-20b")
