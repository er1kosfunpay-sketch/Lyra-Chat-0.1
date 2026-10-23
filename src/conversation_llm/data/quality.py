"""Text quality checks, PII masking and small statistical helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|(?<![\w@])(?:[a-z0-9][a-z0-9-]{0,62}\.)+(?:ru|com|org|net|io|me|tv|info|рф)(?:/\S*)?",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|8|7)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-()]*\d{2}[\s\-()]*\d{2}(?!\d)"
)
_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{7,}", re.DOTALL)
_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{3,}")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_PII_HINT_RE = re.compile(r"[@\d]|https?://|www\.|\b[a-z0-9-]+\.(?:ru|com|org|net|io|me|tv|info)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")

_SPAM_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"переходи(?:те)?\s+по\s+ссылке",
        r"ссылка\s+в\s+(?:профиле|описании)",
        r"подпис(?:ывайся|ывайтесь|шись)\s+на",
        r"заработа(?:ть|й|ю)\s+(?:от|на)\s+\d",
        r"промокод",
        r"казино|букмекер|ставки\s+на\s+спорт",
        r"бинарн(?:ые|ых)\s+опцион",
        r"пирамид[аы]",
        r"инвестиц(?:ии|ия|ионн)",
        r"криптовалют|сигналы\s+по\s+крипт",
        r"удалённ(?:ая|ую)\s+работ(?:а|у)\s+без\s+опыта",
        r"доход\s+от\s+\d+\s*%",
        r"\bviagra\b|\bcasino\b",
    )
]
_SPAM_ANY_RE = re.compile("|".join(f"(?:{pattern.pattern})" for pattern in _SPAM_PATTERNS), re.IGNORECASE)


@dataclass
class PiiCounts:
    urls: int = 0
    emails: int = 0
    phones: int = 0
    mentions: int = 0
    long_numbers: int = 0

    def add(self, other: "PiiCounts") -> None:
        for key in self.__dict__:
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def total(self) -> int:
        return sum(self.__dict__.values())

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def repair_mojibake(text: str) -> str:
    """Repair the two most common UTF-8-as-Latin1/CP1252 corruptions."""
    if not text:
        return text
    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        if encoding == "latin1" and not re.search(r"[ÐÑ][\x80-\xff]", text):
            continue
        if encoding == "cp1252" and "â€" not in text:
            continue
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if len(candidates) == 1:
        return text
    # Prefer the candidate with the most Cyrillic letters and fewest replacement chars.
    return max(candidates, key=lambda value: (len(_CYRILLIC_RE.findall(value)), -value.count("\ufffd")))


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = repair_mojibake(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def mask_pii(text: str, counts: PiiCounts | None = None, enabled: bool = True) -> str:
    if not enabled or not text or not _PII_HINT_RE.search(text):
        return text
    local = PiiCounts()
    text, n = _URL_RE.subn("<ссылка>", text)
    local.urls += n
    text, n = _EMAIL_RE.subn("<почта>", text)
    local.emails += n
    text, n = _PHONE_RE.subn("<телефон>", text)
    local.phones += n
    text, n = _MENTION_RE.subn("<упоминание>", text)
    local.mentions += n
    text, n = _LONG_NUMBER_RE.subn("<номер>", text)
    local.long_numbers += n
    if counts is not None:
        counts.add(local)
    return text


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    if limit < 8:
        return text[:limit], True
    window = text[: limit - 1]
    matches = list(_SENTENCE_END.finditer(window))
    cut = matches[-1].end() if matches and matches[-1].end() >= int(limit * 0.6) else -1
    if cut < 0:
        space = window.rfind(" ")
        cut = space if space >= int(limit * 0.6) else limit - 1
    return window[:cut].rstrip() + "…", True


def is_spam(text: str) -> bool:
    if not _SPAM_ANY_RE.search(text):
        return False
    matches = sum(1 for pattern in _SPAM_PATTERNS if pattern.search(text))
    url_count = len(_URL_RE.findall(text))
    return matches >= 1 and url_count >= 1 or matches >= 2


def check_turn(text: str, min_letters: int = 2) -> tuple[bool, str]:
    """Return (acceptable, reason).  Typos and slang are intentionally kept."""
    if not text:
        return False, "empty"
    if text.count("\ufffd") / max(len(text), 1) > 0.15:
        return False, "broken_encoding"
    text_without_links = _URL_RE.sub(" ", text)
    letters = _LETTER_RE.findall(text_without_links)
    if len(letters) < min_letters:
        return False, "no_letters"
    if _REPEATED_CHAR_RE.search(text):
        return False, "repeated_characters"
    if is_spam(text):
        return False, "spam"
    compact = re.sub(r"\s+", "", text_without_links)
    if len(compact) >= 12 and len(set(compact.lower())) <= 3:
        return False, "no_information"
    if len(compact) >= 400 and len(set(compact)) / len(compact) < 0.03:
        return False, "degenerate"
    return True, "ok"


def ngram_repetition(turns: Sequence[str], n: int = 4) -> float:
    words: list[str] = []
    for turn in turns:
        words.extend(re.findall(r"\w+", turn.lower()))
    if len(words) < n * 3:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) / max(sum(grams.values()), 1)


def word_fingerprint(turns: Sequence[str]) -> str:
    words = [w for turn in turns for w in re.findall(r"\w+", turn.lower())]
    return " ".join(words)


def bag_fingerprint(turns: Sequence[str]) -> str:
    words = [w for turn in turns for w in re.findall(r"\w+", turn.lower())]
    return " ".join(sorted(words))


def letter_count(text: str) -> int:
    return len(_LETTER_RE.findall(text))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - rank) + ordered[high] * (rank - low))


def summarize_lengths(values: Iterable[float]) -> dict[str, float]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": round(sum(data) / len(data), 2),
        "p50": round(percentile(data, 0.50), 2),
        "p90": round(percentile(data, 0.90), 2),
        "p99": round(percentile(data, 0.99), 2),
        "max": round(max(data), 2),
    }


class DialogueDeduplicator:
    """Exact and bag-of-words near-duplicate detection.

    The bag fingerprint removes word-order differences ("кто ты" vs "ты кто").
    It is deliberately cheap enough for millions of dialogues; it is not a
    full MinHash implementation and is documented as such in the data report.
    """

    def __init__(self, near_duplicates: bool = True, min_words_for_near: int = 6) -> None:
        self._exact: set[str] = set()
        self._bags: set[str] = set()
        self._near_duplicates = near_duplicates
        self._min_words = min_words_for_near

    def check(self, turns: Sequence[str]) -> str | None:
        exact = word_fingerprint(turns)
        if not exact:
            return "empty"
        if exact in self._exact:
            return "exact_duplicate"
        self._exact.add(exact)
        if self._near_duplicates:
            words = exact.split()
            if len(words) >= self._min_words:
                bag = bag_fingerprint(turns)
                if bag in self._bags:
                    return "near_duplicate"
                self._bags.add(bag)
        return None

    def __len__(self) -> int:
        return len(self._exact)


@dataclass
class DialogueQualityStats:
    seen: int = 0
    kept: int = 0
    turns_seen: int = 0
    turns_kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    turns_dropped: Counter[str] = field(default_factory=Counter)
    truncated_turns: int = 0
    truncated_dialogues: int = 0
    pii: PiiCounts = field(default_factory=PiiCounts)
    turn_lengths: list[int] = field(default_factory=list)
    dialogue_lengths: list[int] = field(default_factory=list)
    turns_per_dialogue: list[int] = field(default_factory=list)
    repetition_scores: list[float] = field(default_factory=list)
    roles_inferred: int = 0
    roles_provided: int = 0
    role_confidences: list[float] = field(default_factory=list)

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1

    def note_pii(self, counts: PiiCounts) -> None:
        self.pii.add(counts)
