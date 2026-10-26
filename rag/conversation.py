"""Follow-up questions in a chat: rewrite them into standalone questions.

Each question is retrieved on its own, so a follow-up like "And which algorithm?"
or "What was its accuracy?" finds nothing: the words that say what "it" is are in
an earlier turn. Before retrieval, such a question is rewritten by the LLM into a
standalone one ("Which algorithm does the resume screening paper use?"), using the
last few turns of the conversation.

Only questions that look like follow-ups are rewritten, because a rewrite costs an
extra LLM call (a few seconds on a local CPU model). A rewrite that comes back empty
or malformed is discarded and the original question is used.
"""

import re
from collections.abc import Sequence

from rag.llm import LLMClient, Message

Turn = dict[str, str]  # {"question": "...", "answer": "..."}

_REFERS_BACK = re.compile(
    r"\b(?:it|its|they|them|their|theirs|this|that|these|those|he|him|his|she|her|"
    r"the\s+same|former|latter|above|previous|earlier)\b"
    r"|^\s*(?:and|also|what\s+about|how\s+about|and\s+what|so|then|why)\b",
    re.IGNORECASE,
)
SHORT_QUESTION_WORDS = 3
MAX_ANSWER_CHARS = 400   # of each earlier answer, enough to resolve references
MAX_REWRITE_CHARS = 500

REWRITE_SYSTEM_PROMPT = (
    "You rewrite the user's follow-up question into a standalone question that can be "
    "understood without the conversation. Use the conversation only to replace "
    "references such as 'it', 'they' or 'that' with what they refer to. Do not answer "
    "the question. Keep the user's wording otherwise. Output only the rewritten question."
)


def needs_rewrite(question: str, history: Sequence[Turn] | None) -> bool:
    """True when there is a conversation and the question seems to depend on it."""
    if not history:
        return False
    words = question.split()
    return len(words) <= SHORT_QUESTION_WORDS or _REFERS_BACK.search(question) is not None


def build_rewrite_messages(question: str, history: Sequence[Turn]) -> list[Message]:
    lines = []
    for turn in history:
        answer = " ".join(turn.get("answer", "").split())[:MAX_ANSWER_CHARS]
        lines.append(f"User: {turn.get('question', '').strip()}")
        lines.append(f"Assistant: {answer}")
    user = ("Conversation:\n" + "\n".join(lines)
            + f"\n\nFollow-up question: {question.strip()}\nStandalone question:")
    return [{"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"^(?:standalone question|rewritten question|question)\s*:\s*", "", text,
                  flags=re.IGNORECASE)
    return text.strip().strip('"').strip()


def rewrite_question(llm: LLMClient, question: str, history: Sequence[Turn]) -> str:
    """The standalone version of `question`, or `question` itself if the rewrite
    is unusable (empty, too long, or several lines)."""
    reply = llm.complete(build_rewrite_messages(question, history))
    text = _clean(reply.text)
    if not text or len(text) > MAX_REWRITE_CHARS or "\n" in text:
        return question
    return text
