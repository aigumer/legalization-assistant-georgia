"""Ask a question from the terminal, without starting the streamlit.

    uv run python scripts/ask.py "question"
"""

import argparse

from legalization_assistant.config import DEFAULT_NUM_RESULTS
from legalization_assistant.rag import retrieve, stream_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", "--num-results", type=int, default=DEFAULT_NUM_RESULTS)
    parser.add_argument(
        "--no-translate", action="store_true", help="Skip translating non-English questions."
    )
    args = parser.parse_args()

    sources = retrieve(
        args.question, num_results=args.num_results, translate=not args.no_translate
    )
    print("Retrieved:")
    for source in sources:
        print(f"  - {source['article']} - {source['title']}")
    print()

    try:
        for text in stream_answer(args.question, sources):
            print(text, end="", flush=True)
    except RuntimeError as error:
        raise SystemExit(f"\n{error}") from error
    print()


if __name__ == "__main__":
    main()
