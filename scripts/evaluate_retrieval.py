"""Score retrieval against a ground-truth set of question -> article pairs.
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --sweep
"""

import argparse
import csv
from pathlib import Path

from legalization_assistant.config import (
    DATA_DIR,
    DEFAULT_BOOSTS,
    DEFAULT_NUM_RESULTS,
    GROUND_TRUTH_PATH,
)
from legalization_assistant.search import get_search

SEED_PATH = DATA_DIR / "ground_truth_seed.csv"

# Boost configurations compared by --sweep, to show what the defaults buy.
BOOST_VARIANTS: dict[str, dict[str, float] | None] = {
    "uniform (no boosts)": {"title": 1.0, "chapter": 1.0, "section": 1.0, "text": 1.0},
    "title 3 (default)": DEFAULT_BOOSTS,
    "title 3 + chapter 1.5": {"title": 3.0, "chapter": 1.5, "section": 0.5, "text": 1.0},
    "title 8": {"title": 8.0, "chapter": 1.0, "section": 1.0, "text": 1.0},
    "body text only": {"title": 0.0, "chapter": 0.0, "section": 0.0, "text": 1.0},
}


def load_ground_truth(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate(
    ground_truth: list[dict[str, str]],
    num_results: int,
    boosts: dict[str, float] | None,
) -> tuple[float, float]:
    search = get_search()
    hits = 0
    reciprocal_ranks = 0.0

    for row in ground_truth:
        # Scored without the LLM translation step: this measures the index.
        results = search.search(row["question"], num_results=num_results, boosts=boosts)
        # Match on the document too when the ground truth names one, so another
        # law's "Article 15" cannot be counted as a hit. The hand-written seed
        # set predates the column and omits it.
        expected_doc = row.get("doc_id")
        found = [
            (result["article"], result["doc_id"] if expected_doc else None)
            for result in results
        ]
        expected = (row["article"], expected_doc or None)
        if expected in found:
            hits += 1
            reciprocal_ranks += 1.0 / (found.index(expected) + 1)

    total = len(ground_truth) or 1
    return hits / total, reciprocal_ranks / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=GROUND_TRUTH_PATH if GROUND_TRUTH_PATH.exists() else SEED_PATH,
        help="CSV with `question` and `article` columns.",
    )
    parser.add_argument("-k", "--num-results", type=int, default=DEFAULT_NUM_RESULTS)
    parser.add_argument("--sweep", action="store_true", help="Compare boost configurations.")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    print(f"{len(ground_truth)} questions from {args.ground_truth.name}\n")

    if args.sweep:
        print(f"{'configuration':<34} {'hit rate':>10} {'MRR':>8}")
        print("-" * 54)
        for name, boosts in BOOST_VARIANTS.items():
            hit_rate, mrr = evaluate(ground_truth, args.num_results, boosts)
            print(f"{name:<34} {hit_rate:>9.1%} {mrr:>8.3f}")
        return

    hit_rate, mrr = evaluate(ground_truth, args.num_results, DEFAULT_BOOSTS)
    print(f"hit rate @{args.num_results}: {hit_rate:.1%}")
    print(f"MRR      @{args.num_results}: {mrr:.3f}")


if __name__ == "__main__":
    main()
