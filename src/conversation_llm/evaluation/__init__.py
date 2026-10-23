"""Evaluation reports and fixed dialogue prompts."""

from .evaluator import (
    build_eval_loader,
    evaluate_loss,
    generate_scenarios,
    save_evaluation,
    text_quality_metrics,
)
from .prompts import SCENARIOS, EvaluationScenario, scenario_names

__all__ = [
    "build_eval_loader",
    "evaluate_loss",
    "generate_scenarios",
    "save_evaluation",
    "text_quality_metrics",
    "SCENARIOS",
    "EvaluationScenario",
    "scenario_names",
]
