"""Phase 1 — Agentic repair loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from textocad.services.requirement_spec import RequirementSpec
from textocad.services.verification import (
    CheckFailure,
    GeometrySample,
    VerificationReport,
    verify,
)

GenerateFn = Callable[[str, list[dict[str, Any]]], Any]
BuildFn = Callable[[Any], GeometrySample]


@dataclass
class RepairOutcome:
    success: bool
    attempts: int
    artifact: Any = None
    report: VerificationReport | None = None
    honest_message: str = ""
    history: list[VerificationReport] = field(default_factory=list)

    @property
    def shippable(self) -> bool:
        """A build is shippable ONLY if the gate passed. Honest-failure-never-fake."""
        return self.success


def _honest_failure_message(report: VerificationReport, attempts: int) -> str:
    bullets = "\n".join(f"  - {f.message}" for f in report.failures)
    return (
        f"Could not produce an exact model after {attempts} attempt(s). "
        f"The result is NOT exact and was NOT shipped.\n"
        f"What does not match the requirement ({report.understood_spec}):\n{bullets}"
    )


def run_with_repair(
    prompt: str,
    spec: RequirementSpec,
    generate_fn: GenerateFn,
    build_fn: BuildFn,
    max_attempts: int = 3,
) -> RepairOutcome:
    prior: list[dict[str, Any]] = []
    history: list[VerificationReport] = []
    last_report: VerificationReport | None = None

    for attempt in range(1, max_attempts + 1):
        artifact = generate_fn(prompt, prior)
        sample = build_fn(artifact)
        report = verify(sample, spec)
        history.append(report)
        last_report = report

        if report.passed:
            return RepairOutcome(
                success=True, attempts=attempt, artifact=artifact,
                report=report, history=history,
            )


        prior = [f.as_delta() for f in report.failures]

    return RepairOutcome(
        success=False,
        attempts=max_attempts,
        report=last_report,
        history=history,
        honest_message=_honest_failure_message(last_report, max_attempts),
    )
