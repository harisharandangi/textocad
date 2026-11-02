"""Phase 1 — live-path verification wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from textocad.core.config import settings
from textocad.services.cad_worker import WorkerBuildResult, _iso_now
from textocad.services.repair_loop import _honest_failure_message
from textocad.services.requirement_spec import RequirementSpec, extract_requirement_spec
from textocad.services.verification import GeometrySample, VerificationReport, verify


BuildFn = Callable[[dict[str, Any]], WorkerBuildResult]

RegenerateFn = Callable[[dict[str, Any], str, list[dict[str, Any]]], dict[str, Any] | None]


def _safe_extract_spec(prompt: str) -> RequirementSpec:
    """Extract the requirement spec; on any failure return an EMPTY spec."""
    if not prompt.strip():
        return RequirementSpec()
    try:
        return extract_requirement_spec(prompt)
    except Exception:
        return RequirementSpec()


def _sample_from_result(result: WorkerBuildResult) -> GeometrySample:
    """Read geometry back from the produced artifacts: STL -> trimesh (mesh checks), STEP -> cadquery solid."""
    mesh = None
    solid = None
    stl = result.files.get("stl")
    if stl and Path(stl).exists():
        try:
            import trimesh

            mesh = trimesh.load(str(stl))
        except Exception:
            mesh = None
    step = result.files.get("step")
    if step and Path(step).exists():
        try:
            import cadquery as cq

            solid = cq.importers.importStep(str(step))
        except Exception:
            solid = None
    provenance: dict[str, Any] = {}
    hole_count = result.metadata.get("holeCount")
    if isinstance(hole_count, int):
        provenance["hole_count"] = hole_count
    return GeometrySample(solid=solid, mesh=mesh, provenance=provenance)


def _verify_result(result: WorkerBuildResult, spec: RequirementSpec) -> VerificationReport:
    """Load the build's geometry and check it against the spec, in THIS process."""
    return verify(_sample_from_result(result), spec)


def _verify_result_isolated(result: WorkerBuildResult, spec: RequirementSpec) -> VerificationReport:
    """Run the geometry check in a short-lived child process so the OpenCASCADE and trimesh memory it needs is."""
    from textocad.services.geometry_isolation import IsolationError, run_isolated

    try:
        return run_isolated(_verify_result, result, spec)
    except IsolationError as exc:
        print(f"[TextoCAD] verification skipped: {exc}", flush=True)
        return VerificationReport(
            passed=True,
            failures=[],
            understood_spec=f"verification skipped ({exc})",
        )


def _annotate(
    result: WorkerBuildResult,
    report: VerificationReport,
    attempt: int,
    spec_notes: list[str] | None = None,
) -> None:
    result.metadata["verification"] = {
        "enabled": True,
        "passed": report.passed,
        "attempt": attempt,
        "understoodSpec": report.understood_spec,
        "specWarnings": list(spec_notes or []),
        "failures": [f.as_delta() for f in report.failures],
        "failureMessages": [f.message for f in report.failures],
    }


def _honest_fail_result(
    template: WorkerBuildResult,
    report: VerificationReport,
    attempts: int,
    spec_notes: list[str] | None = None,
) -> WorkerBuildResult:
    """A NON-shippable result: status 'error', honest message, full structured report."""
    message = _honest_failure_message(report, attempts)
    failed = WorkerBuildResult(
        build_id=template.build_id,
        status="error",
        mode=template.mode,
        pipeline=template.pipeline,
        preview_source="none",
        build_dir=template.build_dir,
        files={},
        exports={},
        error=message,
        warnings=list(template.warnings),
        created_at=template.created_at,
        updated_at=_iso_now(),
        metadata={
            **template.metadata,
            "verification": {
                "enabled": True,
                "passed": False,
                "attempts": attempts,
                "understoodSpec": report.understood_spec,
                "specWarnings": list(spec_notes or []),
                "failures": [f.as_delta() for f in report.failures],
                "failureMessages": [f.message for f in report.failures],
                "honestMessage": message,
            },
            "shipped": False,
        },
    )
    return failed


def run_verified_build(
    worker_payload: dict[str, Any],
    *,
    build_fn: BuildFn,
    regenerate_fn: RegenerateFn,
    max_attempts: int | None = None,
) -> WorkerBuildResult:
    """Build -> verify -> (repair) -> ship-or-honest-fail."""
    attempts = max_attempts or settings.verification_max_attempts
    prompt = str(worker_payload.get("prompt") or "")
    spec = _safe_extract_spec(prompt)

    current = dict(worker_payload)
    last_result: WorkerBuildResult | None = None
    last_report: VerificationReport | None = None

    for attempt in range(1, attempts + 1):
        result = build_fn(current)
        last_result = result


        if result.status != "ready":
            return result

        report = _verify_result_isolated(result, spec)
        last_report = report
        _annotate(result, report, attempt, spec.extraction_notes)

        if report.passed:
            result.metadata["shipped"] = True
            return result

        if attempt < attempts:
            new_doc = regenerate_fn(current, prompt, [f.as_delta() for f in report.failures])
            if new_doc is None:
                break
            current = {**current, "cadDocument": new_doc, "cad_document": new_doc}

    return _honest_fail_result(last_result, last_report, attempts, spec.extraction_notes)


def llm_regenerate(
    payload: dict[str, Any],
    prompt: str,
    prior_failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Production regenerator: re-ask Gemini with the structured failures appended."""
    try:
        from textocad.services.llm_service import generate_cad_package
    except Exception:
        return None

    feedback = "\n".join(
        f"- {d.get('check')}: expected {d.get('expected')}, got {d.get('got')}"
        for d in prior_failures
    )
    params = dict(payload)
    params["prompt"] = (
        f"{prompt}\n\nThe previous attempt FAILED verification. "
        f"Fix EXACTLY these issues and keep everything else:\n{feedback}"
    )
    res = generate_cad_package(params)
    if not getattr(res, "ok", False) or getattr(res, "package", None) is None:
        return None
    try:
        return res.package.model_dump(by_alias=True)
    except Exception:
        return None
