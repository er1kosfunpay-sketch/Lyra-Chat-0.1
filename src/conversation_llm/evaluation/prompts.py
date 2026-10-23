"""Fixed dialogue scenarios for manual/qualitative evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationScenario:
    name: str
    tags: list[str]
    messages: list[str] = field(default_factory=list)


SCENARIOS: list[EvaluationScenario] = [
    EvaluationScenario("greeting", ["greeting"], ["Привет! Как дела?"]),
    EvaluationScenario("short_question", ["short"], ["Что делаешь?"]),
    EvaluationScenario("slang", ["slang", "short"], ["Здарова, чё как?"]),
    EvaluationScenario(
        "clarification",
        ["clarification", "context"],
        ["Я вчера начал смотреть новый сериал.", "Как думаешь, стоит его досматривать?"],
    ),
    EvaluationScenario(
        "topic_change",
        ["context", "topic_change"],
        ["Привет, как прошли выходные?", "Кстати, ты любишь кофе?"],
    ),
    EvaluationScenario(
        "shorter",
        ["instruction", "length"],
        ["Расскажи, чем ты обычно занимаешься вечером.", "Покороче, пожалуйста."],
    ),
    EvaluationScenario(
        "emotional",
        ["emotion"],
        ["У меня сегодня был ужасный день, всё валится из рук."],
    ),
    EvaluationScenario(
        "memory",
        ["context", "memory"],
        ["Меня зовут Аня.", "Как меня зовут?"],
    ),
    EvaluationScenario(
        "smalltalk",
        ["context", "multi_turn"],
        ["Привет.", "Как настроение?", "Что нового?"],
    ),
    EvaluationScenario(
        "open",
        ["long"],
        ["Расскажи что-нибудь интересное."],
    ),
]


def scenario_names() -> list[str]:
    return [scenario.name for scenario in SCENARIOS]
