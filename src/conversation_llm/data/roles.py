"""Speaker-role reconstruction.

The source corpus stores ordered messages without speaker identifiers.  We do
not pretend that these roles are ground truth.  The only defensible default is
to alternate starting with the human ("user"), record that the labels were
inferred, and attach a heuristic confidence for analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_GREETING_RE = re.compile(
    r"^\s*(?:привет|прив|здравствуй|здравствуйте|добрый (?:день|вечер)|доброе утро|хай|ку|hello|hi)\b",
    re.IGNORECASE,
)
_ANSWER_START_RE = re.compile(
    r"^\s*(?:да|нет|не|ага|угу|конечно|возможно|наверное|думаю|согласен|согласна|"
    r"почему|потому|это|я|у меня|мне|ну|хорошо|ладно)\b",
    re.IGNORECASE,
)
_ROLE_MAP = {
    "user": "user",
    "human": "user",
    "пользователь": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "ai": "assistant",
    "ассистент": "assistant",
    "system": "system",
    "system_prompt": "system",
}


@dataclass
class RoleAssignment:
    roles: list[str]
    inferred: bool
    confidence: float
    user_question_rate: float = 0.0
    assistant_question_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "roles": list(self.roles),
            "inferred": self.inferred,
            "confidence": round(self.confidence, 4),
            "user_question_rate": round(self.user_question_rate, 4),
            "assistant_question_rate": round(self.assistant_question_rate, 4),
        }


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def infer_roles(turns: Sequence[str], provided: Sequence[str] | None = None) -> RoleAssignment:
    if len(turns) == 0:
        raise ValueError("Cannot infer roles for an empty dialogue")

    if provided is not None:
        if len(provided) != len(turns):
            raise ValueError("Role list length must match the number of turns")
        mapped: list[str] = []
        for role in provided:
            normalized = _ROLE_MAP.get(role.strip().lower())
            if normalized is None:
                raise ValueError(f"Unknown speaker role: {role!r}")
            mapped.append(normalized)
        if not any(role == "user" for role in mapped):
            raise ValueError("Dialogue has no user turns")
        return RoleAssignment(roles=mapped, inferred=False, confidence=1.0)

    roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(turns))]
    user_questions = [turns[i].strip().endswith("?") for i in range(0, len(turns), 2)]
    assistant_questions = [turns[i].strip().endswith("?") for i in range(1, len(turns), 2)]
    user_q = _rate(user_questions)
    assistant_q = _rate(assistant_questions)

    answer_starts = [
        bool(_ANSWER_START_RE.match(turns[i])) for i in range(1, len(turns), 2)
    ]
    user_answer_starts = [
        bool(_ANSWER_START_RE.match(turns[i])) for i in range(2, len(turns), 2)
    ]
    greeting = 1.0 if _GREETING_RE.match(turns[0]) else 0.0

    score = 0.5
    score += 0.15 * (user_q - assistant_q)
    score += 0.10 * greeting
    score += 0.10 * (_rate(answer_starts) - _rate(user_answer_starts))
    if len(turns) >= 1 and turns[0].strip().endswith("?"):
        score += 0.03
    confidence = max(0.5, min(0.85, score))
    return RoleAssignment(
        roles=roles,
        inferred=True,
        confidence=confidence,
        user_question_rate=user_q,
        assistant_question_rate=assistant_q,
    )
