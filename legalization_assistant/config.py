"""Shared paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
GROUND_TRUTH_PATH = PROCESSED_DIR / "ground_truth.csv"

# Articles longer than this are split into parts on paragraph boundaries.
MAX_CHUNK_CHARS = 2000

DEFAULT_NUM_RESULTS = 5
# Parts of one article that may occupy result slots, so a long article cannot
# crowd out the others.
MAX_PARTS_PER_ARTICLE = 2
OVERFETCH_FACTOR = 4
DEFAULT_BOOSTS = {
    "title": 3.0,
    "chapter": 1.0,
    "section": 1.0,
    "text": 1.0,
}

MODEL = os.getenv("LEGAL_ASSISTANT_MODEL", "openai/gpt-oss-120b")
# Groq counts `max_completion_tokens` in full against the tokens-per-minute
# quota, used or not, so prompt + answer budget has to fit inside one minute's
# worth. Every limit below is derived from that, and `fit_context` and
# `trim_history` in rag.py hold the prompt to it.
TOKENS_PER_MINUTE = int(os.getenv("LEGAL_ASSISTANT_TOKENS_PER_MINUTE", "8000"))
MAX_TOKENS = 3000
MAX_SOURCE_CHARS = 2400
REASONING_EFFORT = os.getenv("LEGAL_ASSISTANT_REASONING_EFFORT", "medium")

# Rough English average, and deliberately low: the budget guard should
# over-estimate the prompt rather than under-estimate it.
CHARS_PER_TOKEN = 4
# System prompt, the question, and the per-excerpt headers in format_context.
PROMPT_OVERHEAD_TOKENS = 700
# Prior turns are replayed on every follow-up, so without a cap the prompt grows
# until the quota rejects it. Oldest turns are dropped first.
MAX_HISTORY_CHARS = 4000
# What is left for retrieved excerpts once everything above is reserved.
MAX_CONTEXT_CHARS = (
    TOKENS_PER_MINUTE - MAX_TOKENS - PROMPT_OVERHEAD_TOKENS
) * CHARS_PER_TOKEN - MAX_HISTORY_CHARS

TRANSLATION_MODEL = os.getenv("LEGAL_ASSISTANT_TRANSLATION_MODEL", "openai/gpt-oss-20b")
