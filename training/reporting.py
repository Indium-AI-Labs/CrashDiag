"""Dependency-free training and evaluation reports for CrashDiag.

Reports visualize numeric values already emitted by Transformers/TRL or the
mechanical evaluator. They never call a model, reconstruct a reward, or decide
whether a fault was fixed. SVG output keeps report generation usable in tests,
CPU jobs, and Kaggle without adding a plotting dependency.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReportingError(RuntimeError):
    """Raised when recorded metrics cannot produce a truthful report."""


@dataclass(frozen=True)
class ReportBundle:
    """Inventory returned after report files are written."""

    report_type: str
    charts: tuple[Path, ...]
    metrics_path: Path
    summary_path: Path
    markdown_path: Path
    summary: Mapping[str, Any]

    @property
    def files(self) -> tuple[Path, ...]:
        return (
            self.metrics_path,
            self.summary_path,
            self.markdown_path,
            *self.charts,
        )


_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ReportingError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportingError(f"invalid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ReportingError(f"{label} must contain a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _format_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude and (magnitude < 0.001 or magnitude >= 10000):
        return f"{value:.3e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _normalize_history(
    trainer_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[float, float]]]]:
    history = trainer_state.get("log_history")
    if not isinstance(history, list) or not history:
        raise ReportingError("trainer_state.json has no non-empty log_history")

    records: list[dict[str, Any]] = []
    # A final evaluation can log the same metric at the same step twice. Keep
    # the latest value rather than drawing a misleading vertical segment.
    latest: dict[str, dict[float, float]] = {}
    for index, raw_record in enumerate(history):
        if not isinstance(raw_record, Mapping):
            continue
        step_value = _number(raw_record.get("step"))
        step = step_value if step_value is not None else float(index)
        epoch = _number(raw_record.get("epoch"))
        metrics: dict[str, float] = {}
        for key, value in raw_record.items():
            name = str(key)
            if name in {"step", "epoch"}:
                continue
            numeric = _number(value)
            if numeric is None:
                continue
            metrics[name] = numeric
            latest.setdefault(name, {})[step] = numeric
        if metrics:
            records.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "metrics": dict(sorted(metrics.items())),
                }
            )

    if not records or not latest:
        raise ReportingError("trainer_state.json contains no finite numeric metrics")
    series = {
        name: sorted(points.items())
        for name, points in sorted(latest.items())
        if points
    }
    return records, series


def _metric_summary(
    series: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, points in series.items():
        values = [value for _, value in points]
        summary[name] = {
            "points": len(values),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
        }
    return summary


def _sample_points(
    points: Sequence[tuple[float, float]], max_points: int = 1200
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return list(points)
    stride = math.ceil(len(points) / max_points)
    sampled = list(points[::stride])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _line_chart(
    path: Path,
    *,
    title: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
) -> Path:
    plotted = {
        name: _sample_points(points)
        for name, points in list(series.items())[: len(_COLORS)]
        if points
    }
    if not plotted:
        raise ReportingError(f"chart {title!r} has no numeric series")

    width, height = 960, 520
    left, right, top, bottom = 82, 28, 100, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for points in plotted.values() for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.5)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
            f'font-family="sans-serif" font-size="22" font-weight="600">'
            f"{html.escape(title)}</text>"
        ),
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        parts.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
                    f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="monospace" font-size="12" fill="#4b5563">'
                    f"{html.escape(_format_number(value))}</text>"
                ),
            ]
        )
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        value = x_min + fraction * (x_max - x_min)
        parts.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                    f'y2="{height - bottom}" stroke="#f3f4f6" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x:.2f}" y="{height - bottom + 22}" '
                    f'text-anchor="middle" font-family="monospace" font-size="12" '
                    f'fill="#4b5563">{html.escape(_format_number(value))}</text>'
                ),
            ]
        )
    parts.extend(
        [
            (
                f'<line x1="{left}" y1="{top}" x2="{left}" '
                f'y2="{height - bottom}" stroke="#111827" stroke-width="1.5"/>'
            ),
            (
                f'<line x1="{left}" y1="{height - bottom}" '
                f'x2="{width - right}" y2="{height - bottom}" '
                f'stroke="#111827" stroke-width="1.5"/>'
            ),
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 18}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="14">'
                "Training step</text>"
            ),
        ]
    )

    for index, (name, points) in enumerate(plotted.items()):
        color = _COLORS[index]
        coordinates = " ".join(
            f"{x_position(x):.2f},{y_position(y):.2f}" for x, y in points
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if len(points) <= 60:
            parts.extend(
                f'<circle cx="{x_position(x):.2f}" cy="{y_position(y):.2f}" '
                f'r="3" fill="{color}"/>'
                for x, y in points
            )
        label = html.escape(name)
        legend_x = left + (index % 3) * (plot_width / 3)
        legend_y = 55 + (index // 3) * 20
        parts.extend(
            [
                (
                    f'<line x1="{legend_x:.2f}" y1="{legend_y}" '
                    f'x2="{legend_x + 24:.2f}" y2="{legend_y}" '
                    f'stroke="{color}" stroke-width="3"/>'
                ),
                (
                    f'<text x="{legend_x + 30:.2f}" y="{legend_y + 4}" '
                    f'font-family="sans-serif" '
                    f'font-size="12" fill="#111827">{label}</text>'
                ),
            ]
        )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _training_groups(
    kind: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    report_title: str,
) -> list[tuple[str, str, list[str]]]:
    keys = list(series)
    loss = [key for key in keys if key == "loss" or key.endswith("_loss")]
    learning_rate = [key for key in keys if "learning_rate" in key]
    gradient = [key for key in keys if "grad_norm" in key]
    groups: list[tuple[str, str, list[str]]] = [
        ("loss.svg", f"{report_title} — loss", loss[:6]),
        (
            "learning_rate.svg",
            f"{report_title} — learning rate",
            learning_rate[:6],
        ),
        ("gradient_norm.svg", f"{report_title} — gradient norm", gradient[:6]),
    ]
    if kind == "grpo":
        rewards = [
            key
            for key in keys
            if "reward" in key.lower() or "success_rate" in key.lower()
        ]
        diagnostics = [
            key
            for key in keys
            if any(
                marker in key.lower()
                for marker in (
                    "entropy",
                    "kl",
                    "completion",
                    "clip_ratio",
                    "frac_reward_zero_std",
                )
            )
        ]
        groups.extend(
            [
                (
                    "reward.svg",
                    f"{report_title} — mechanical reward metrics",
                    rewards[:6],
                ),
                (
                    "policy_diagnostics.svg",
                    f"{report_title} — policy diagnostics",
                    diagnostics[:6],
                ),
            ]
        )
    return [group for group in groups if group[2]]


def _training_markdown(
    *, title: str, source: Path, summary: Mapping[str, Any]
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Source: `{source}`",
        "",
        "These charts visualize recorded trainer metrics. They are diagnostic ",
        "only and do not grade whether infrastructure faults were resolved.",
        "",
        "| Metric | Points | First | Last | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        for name, raw_values in metrics.items():
            if not isinstance(raw_values, Mapping):
                continue
            lines.append(
                "| "
                + str(name).replace("|", "\\|")
                + " | "
                + str(raw_values.get("points", ""))
                + " | "
                + " | ".join(
                    _format_number(float(raw_values[field]))
                    for field in ("first", "last", "min", "max")
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def generate_trainer_report(
    trainer_state_path: str | Path,
    report_directory: str | Path,
    *,
    kind: str,
    title: str | None = None,
) -> ReportBundle:
    """Create strict JSON, Markdown, and SVGs from recorded trainer metrics."""

    normalized_kind = kind.strip().lower()
    if normalized_kind not in {"sft", "grpo"}:
        raise ReportingError("trainer report kind must be 'sft' or 'grpo'")
    state_path = Path(trainer_state_path)
    output = Path(report_directory)
    state = _read_mapping(state_path, "trainer state")
    records, series = _normalize_history(state)
    metric_summary = _metric_summary(series)
    report_title = title or f"CrashDiag {normalized_kind.upper()} training report"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "report_type": f"{normalized_kind}_training",
        "title": report_title,
        "source": state_path.name,
        "record_count": len(records),
        "metrics": metric_summary,
        "scoring": "diagnostic_trainer_metrics_only",
    }
    metrics_path = _write_json(
        output / "metrics_history.json",
        {
            "schema_version": 1,
            "report_type": f"{normalized_kind}_training_history",
            "records": records,
        },
    )
    summary_path = _write_json(output / "metrics_summary.json", summary)
    charts: list[Path] = []
    for filename, chart_title, keys in _training_groups(
        normalized_kind, series, report_title
    ):
        charts.append(
            _line_chart(
                output / filename,
                title=chart_title,
                series={key: series[key] for key in keys},
            )
        )
    if not charts:
        fallback = sorted(
            series,
            key=lambda key: (-len(series[key]), key),
        )[:6]
        charts.append(
            _line_chart(
                output / "recorded_metrics.svg",
                title=f"{normalized_kind.upper()} recorded metrics",
                series={key: series[key] for key in fallback},
            )
        )
    markdown_path = output / "report.md"
    markdown_path.write_text(
        _training_markdown(title=report_title, source=state_path, summary=summary),
        encoding="utf-8",
    )
    return ReportBundle(
        report_type=f"{normalized_kind}_training",
        charts=tuple(charts),
        metrics_path=metrics_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        summary=summary,
    )


def _evaluation_chart(
    path: Path,
    rates: Sequence[tuple[str, float]],
    *,
    title: str,
) -> Path:
    if not rates:
        raise ReportingError("evaluation report contains no per-fault metrics")
    width, height = 960, 540
    left, right, top, bottom = 76, 28, 70, 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    slot = plot_width / len(rates)
    bar_width = slot * 0.62
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
            f'font-family="sans-serif" font-size="22" font-weight="600">'
            f"{html.escape(title)}</text>"
        ),
    ]
    for tick in range(6):
        rate = tick / 5
        y = top + (1 - rate) * plot_height
        parts.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
                    f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="12" fill="#4b5563">'
                    f"{rate:.0%}</text>"
                ),
            ]
        )
    for index, (name, rate) in enumerate(rates):
        x = left + index * slot + (slot - bar_width) / 2
        bar_height = rate * plot_height
        y = top + plot_height - bar_height
        center = x + bar_width / 2
        parts.extend(
            [
                (
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{bar_height:.2f}" fill="{_COLORS[index % len(_COLORS)]}" '
                    'rx="3"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{max(top + 14, y - 8):.2f}" '
                    f'text-anchor="middle" font-family="sans-serif" font-size="12" '
                    f'font-weight="600">{rate:.1%}</text>'
                ),
                (
                    f'<text x="{center:.2f}" y="{height - bottom + 22}" '
                    f'text-anchor="end" transform="rotate(-35 {center:.2f} '
                    f'{height - bottom + 22})" font-family="sans-serif" '
                    f'font-size="12">{html.escape(name)}</text>'
                ),
            ]
        )
    parts.extend(
        [
            (
                f'<line x1="{left}" y1="{top}" x2="{left}" '
                f'y2="{height - bottom}" stroke="#111827" stroke-width="1.5"/>'
            ),
            (
                f'<line x1="{left}" y1="{height - bottom}" '
                f'x2="{width - right}" y2="{height - bottom}" '
                f'stroke="#111827" stroke-width="1.5"/>'
            ),
            "</svg>\n",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def generate_evaluation_report(
    evaluation_path: str | Path,
    report_directory: str | Path,
    *,
    title: str = "CrashDiag held-out mechanical success by fault",
) -> ReportBundle:
    """Visualize success rates already computed by the mechanical evaluator."""

    source_path = Path(evaluation_path)
    output = Path(report_directory)
    evaluation = _read_mapping(source_path, "evaluation report")
    raw_summary = evaluation.get("summary")
    raw_per_fault = evaluation.get("per_fault")
    if not isinstance(raw_summary, Mapping) or not isinstance(raw_per_fault, Mapping):
        raise ReportingError("evaluation report is missing summary or per_fault")

    rates: list[tuple[str, float]] = []
    normalized_faults: dict[str, dict[str, Any]] = {}
    for name in sorted(str(key) for key in raw_per_fault):
        raw_metrics = raw_per_fault.get(name)
        if not isinstance(raw_metrics, Mapping):
            raise ReportingError(f"invalid evaluation metrics for fault {name!r}")
        rate = _number(raw_metrics.get("success_rate"))
        episodes = raw_metrics.get("episodes")
        resolved = raw_metrics.get("resolved")
        if (
            rate is None
            or not 0 <= rate <= 1
            or isinstance(episodes, bool)
            or not isinstance(episodes, int)
            or isinstance(resolved, bool)
            or not isinstance(resolved, int)
            or episodes < 1
            or not 0 <= resolved <= episodes
        ):
            raise ReportingError(f"invalid evaluation metrics for fault {name!r}")
        rates.append((name, rate))
        normalized_faults[name] = {
            "episodes": episodes,
            "resolved": resolved,
            "success_rate": rate,
        }

    overall_rate = _number(raw_summary.get("success_rate"))
    if overall_rate is None or not 0 <= overall_rate <= 1:
        raise ReportingError("evaluation summary has an invalid success_rate")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "mechanical_evaluation",
        "title": title,
        "source": source_path.name,
        "overall_success_rate": overall_rate,
        "per_fault": normalized_faults,
        "scoring": "mechanical_fault_resolution",
    }
    metrics_path = _write_json(
        output / "mechanical_evaluation_metrics.json",
        {
            "schema_version": 1,
            "summary": dict(raw_summary),
            "per_fault": normalized_faults,
        },
    )
    summary_path = _write_json(
        output / "mechanical_evaluation_summary.json", summary
    )
    chart = _evaluation_chart(
        output / "mechanical_success_by_fault.svg",
        rates,
        title=title,
    )
    markdown_lines = [
        f"# {title}",
        "",
        f"Overall mechanically verified success: **{overall_rate:.1%}**",
        "",
        "| Fault | Resolved | Episodes | Success rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in normalized_faults.items():
        escaped_name = name.replace("|", "\\|")
        markdown_lines.append(
            f"| {escaped_name} | {metrics['resolved']} | "
            f"{metrics['episodes']} | {metrics['success_rate']:.1%} |"
        )
    markdown_lines.extend(
        [
            "",
            "Success values come from executable sandbox state, not an LLM grader.",
        ]
    )
    markdown_path = output / "mechanical_evaluation_report.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return ReportBundle(
        report_type="mechanical_evaluation",
        charts=(chart,),
        metrics_path=metrics_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
        summary=summary,
    )


__all__ = [
    "ReportBundle",
    "ReportingError",
    "generate_evaluation_report",
    "generate_trainer_report",
]
