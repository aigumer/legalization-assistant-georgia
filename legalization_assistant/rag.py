"""The RAG flow: retrieve article chunks, then answer strictly from them."""

import os
import re
from collections.abc import Iterator
from functools import lru_cache

import groq

from .config import (
    DEFAULT_NUM_RESULTS,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_CHARS,
    MAX_SOURCE_CHARS,
    MAX_TOKENS,
    MODEL,
    REASONING_EFFORT,
    TRANSLATION_MODEL,
)
from .search import get_search

SYSTEM_PROMPT = """\
You are an assistant that answers questions about visas, residence permits, and \
the legal status of foreigners in Georgia (the country). You answer strictly from \
excerpts of Georgian legislation supplied with each question.

Rules:
- Use only the supplied excerpts. Do not rely on outside knowledge of Georgian or \
any other country's immigration rules, and never invent article numbers, \
timeframes, fees, or document names.
- Cite the article you are relying on inline, e.g. "(Article 15)". When several \
articles apply, cite each where it is used.
- If the excerpts do not answer the question, say so plainly and name what is \
missing. If they answer only part of it, answer that part and flag the rest. Do \
not pad a thin answer with generalities.
- The law states general rules and then exceptions. When an excerpt contains \
conditions, exceptions, or cross-references to other articles, surface them \
rather than presenting the general rule alone.
- Answer in the language the user wrote in.
- Be concrete and brief: lead with the answer, then the conditions that qualify it.
- Write plain prose and short lists. Use a table only when the answer really is \
tabular, and keep emphasis markers out of quoted statutory wording.
- Close with a one-line note that this is information drawn from the legislation, \
not legal advice, and that procedures change — the reader should confirm with the \
Public Service Development Agency or a qualified lawyer.
"""

QUESTION_TEMPLATE = """\
Excerpts from Georgian legislation:

<excerpts>
{context}
</excerpts>

Question: {question}"""


def fit_context(sources: list[dict], budget_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    """Keep the highest-ranked excerpts that fit the prompt's share of the quota.

    Excerpts are ordered by relevance, so the ones dropped are the weakest. This
    is what makes a large `num_results` safe: the request stays inside the quota
    instead of being rejected once the retrieved text is long enough.
    """
    kept: list[dict] = []
    size = 0
    for source in sources:
        length = min(len(source["text"]), MAX_SOURCE_CHARS)
        if kept and size + length > budget_chars:
            break
        kept.append(source)
        size += length
    return kept


def trim_history(history: list[dict], budget_chars: int = MAX_HISTORY_CHARS) -> list[dict]:
    """Keep the most recent whole turns that fit ``budget_chars``.

    Dropping the oldest turns first keeps the follow-up context that matters
    while stopping the prompt from growing with every message.
    """
    kept: list[dict] = []
    size = 0
    for message in reversed(history):
        size += len(message["content"])
        if size > budget_chars:
            break
        kept.append(message)
    kept.reverse()
    if kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def format_context(sources: list[dict]) -> str:
    """Render retrieved chunks so the model can attribute each claim to an article."""
    blocks = []
    for source in sources:
        header = f"[{source['article']} - {source['title']}]"
        if source.get("n_parts", 1) > 1:
            header += f" (part {source['part']} of {source['n_parts']})"
        lines = [
            header,
            f"Source: {source['doc_title']}",
            f"Location: {source['section']} / {source['chapter']}",
        ]
        if source.get("amendments"):
            lines.append(f"Last amended: {source['amendments'][-1]}")
        text = source["text"]
        if len(text) > MAX_SOURCE_CHARS:
            text = f"{text[:MAX_SOURCE_CHARS]} [...excerpt truncated]"
        lines.append(text)
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


NON_LATIN_RE = re.compile(r"[^\x00-\x7f]")

TRANSLATION_PROMPT = """\
Rewrite the user's question as an English keyword query for a search index over \
Georgian immigration legislation.

The output must be in English, whatever language the question is written in - the \
index contains only English text, so any word left in the original language \
matches nothing.

Cover what the user actually asked and nothing more: do not add topics, synonyms, \
or legal terms they did not mention, because every extra term pulls in unrelated \
articles. Where an everyday term has a formal legal equivalent, use it.

Reply with the English query terms only: no notes, no explanation, no quotation \
marks."""


def prepare_query(question: str, translate: bool = True) -> str:
    """Translate non-Latin queries to English so the keyword index can match them."""
    if not translate or not NON_LATIN_RE.search(question):
        return question
    try:
        response = get_client().chat.completions.create(
            model=TRANSLATION_MODEL,
            max_completion_tokens=500,
            reasoning_effort="low",
            temperature=0,
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": question},
            ],
        )
    except groq.APIStatusError as error:
        raise _friendly(error) from error
    except groq.GroqError as error:
        raise RuntimeError(
            f"Could not translate the question into English for search: {error}"
        ) from error
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(
            "Could not translate the question into English for search: the "
            "translation model ran out of its token budget before producing "
            "any output."
        )
    translated = choice.message.content or ""
    return translated.strip() or question


def retrieve(
    question: str,
    num_results: int = DEFAULT_NUM_RESULTS,
    boosts: dict[str, float] | None = None,
    doc_id: str | None = None,
    translate: bool = True,
) -> list[dict]:
    query = prepare_query(question, translate=translate)
    results = get_search().search(
        query, num_results=num_results, boosts=boosts, doc_id=doc_id
    )
    return fit_context(results)


def build_messages(
    question: str,
    sources: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    """Prior turns as plain text, with the excerpts attached to the current question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trim_history(history or []))
    messages.append(
        {
            "role": "user",
            "content": QUESTION_TEMPLATE.format(
                context=format_context(sources) or "(no matching excerpts were found)",
                question=question,
            ),
        }
    )
    return messages


CREDENTIALS_HELP = (
    "No Groq API key found. Set GROQ_API_KEY in your environment or in a .env "
    "file at the project root. Keys are issued at https://console.groq.com/keys."
)


@lru_cache(maxsize=1)
def get_client() -> groq.Groq:
    """The process-wide client, or a clear error about credentials.

    A missing key raises rather than being cached, so setting the key later in
    the same process is picked up on the next call.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(CREDENTIALS_HELP)
    return groq.Groq()


def _friendly(error: groq.APIStatusError) -> RuntimeError:
    """Turn an API failure into something a user of the app can act on."""
    if error.status_code in (413, 429):
        return RuntimeError(
            "Groq rate limit reached - the free tier allows 8,000 tokens per minute, "
            "and this request (prompt plus answer budget) exceeded what is left. "
            "Wait a minute, or lower 'Article excerpts to retrieve' in the sidebar."
        )
    return RuntimeError(f"Groq API error {error.status_code}: {error.message}")


def stream_answer(
    question: str,
    sources: list[dict],
    history: list[dict] | None = None,
    model: str = MODEL,
) -> Iterator[str]:
    """Yield answer text as it is generated.

    gpt-oss returns its reasoning in a separate `reasoning` field, so `content`
    deltas need no filtering.
    """
    try:
        stream = get_client().chat.completions.create(
            model=model,
            max_completion_tokens=MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            messages=build_messages(question, sources, history),
            stream=True,
        )
    except groq.APIStatusError as error:
        raise _friendly(error) from error

    finish_reason = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.content:
            yield choice.delta.content
        finish_reason = choice.finish_reason or finish_reason

    if finish_reason == "length":
        yield "\n\n_(cut off at the token limit - ask a narrower question for the full answer)_"
