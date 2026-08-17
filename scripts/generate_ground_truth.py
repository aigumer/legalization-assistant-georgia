"""Generate an evaluation set by asking the model what each article answers.
"""

import argparse
import csv
import json
from pathlib import Path

import groq

from legalization_assistant.config import GROUND_TRUTH_PATH, MODEL
from legalization_assistant.ingest import load_corpus
from legalization_assistant.rag import get_client

SYSTEM_PROMPT = """\
You write evaluation questions for a search engine over Georgian immigration law.

Given one article, write questions that a foreigner living in or moving to Georgia \
would plausibly type, and that this article answers. Use everyday wording, not \
statutory phrasing, and never mention the article number. Each question must stand \
on its own and be specific enough that this article - not a neighbouring one - is \
the right answer.

Return a JSON array of strings and nothing else."""

USER_TEMPLATE = """\
{article} - {title}
Chapter: {chapter}

{text}

Write exactly {count} questions."""


def generate_for_article(
    client: groq.Groq, document: dict, count: int, model: str
) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=2000,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    article=document["article"],
                    title=document["title"],
                    chapter=document["chapter"],
                    text=document["text"][:4000],
                    count=count,
                ),
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        questions = json.loads(text)
    except json.JSONDecodeError:
        print(f"  ! could not parse response for {document['article']}")
        return []
    return [str(question) for question in questions if str(question).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-per-article", type=int, default=3)
    parser.add_argument("--model", default=MODEL, help="Generation model.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N articles.")
    parser.add_argument("--output", type=Path, default=GROUND_TRUTH_PATH)
    args = parser.parse_args()

    documents: dict[tuple[str, str], dict] = {}
    for document in load_corpus():
        documents.setdefault((document["doc_id"], document["article"]), document)

    articles = list(documents.values())[: args.limit]
    client = get_client()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "doc_id", "article"])
        for position, document in enumerate(articles, start=1):
            print(
                f"[{position}/{len(articles)}] {document['article']} - "
                f"{document['title'][:50]}"
            )
            try:
                questions = generate_for_article(
                    client, document, args.questions_per_article, args.model
                )
            except groq.GroqError as error:
                print(f"  ! stopping after {written} questions: {error}")
                break
            for question in questions:
                writer.writerow([question, document["doc_id"], document["article"]])
                written += 1
            handle.flush()
    print(f"\nWrote {written} questions to {args.output}")


if __name__ == "__main__":
    main()
