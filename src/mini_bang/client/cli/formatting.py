from __future__ import annotations

import json
from typing import Any

from mini_bang.client.cli.errors import CheckResult


def render_checks_text(checks: list[CheckResult]) -> str:
    return "\n".join(f"{item.status} {item.message}" for item in checks)

def render_task_show_text(data: dict[str, Any]) -> str:
    caps = data["public_api_capabilities"]
    return "\n".join(
        [
            f"task_id: {data['task_id']}",
            f"description: {data['description']}",
            f"available_instances: {', '.join(data['available_instances'])}",
            f"default_instance_id: {data['default_instance_id']}",
            f"selected_instance_id: {data['selected_instance_id']}",
            f"max_generate_calls: {data['budget_limits']['max_generate_calls']}",
            f"allowed_extras: {', '.join(data['allowed_extras'])}",
            f"allowed_request_fields: {', '.join(caps.get('allowed_request_fields', []))}",
            f"scorer_class: {data['scorer_class']}",
            f"policy_class: {data['policy_class']}",
            f"gold_artifact_exists: {data['gold_artifact_exists']}",
            "instructions:",
            data.get("instructions") or "",
        ]
    )

def render_run_summary_text(output: dict[str, Any], score_summary: dict[str, Any]) -> str:
    details = score_summary.get("details")
    lines = [
        f"run: {output.get('run_name', output['run_id'])}",
        f"task_id: {output['task_id']}",
        f"instance_id: {output['instance_id']}",
        f"score: {output['score']}",
        f"success: {output['success']}",
        f"run_record_path: {output['run_record_path']}",
    ]
    if details:
        lines.append(f"details: {details}")
    return "\n".join(lines)

def render_result_show_text(data: dict[str, Any], *, include_trace: bool) -> str:
    lines = [
        f"run: {data.get('run_name', data['run_id'])}",
        f"task_id: {data['task_id']}",
        f"instance_id: {data['instance_id']}",
        f"success: {data['score_summary'].get('success')}",
        f"score: {score_label(data['score_summary'])}",
        f"run_record_path: {data['run_record_path']}",
        f"timing: {json.dumps(data['timing'], sort_keys=True)}",
        f"resource_usage: {json.dumps(data['resource_usage'], sort_keys=True)}",
        f"artifacts: {len(data['artifact_refs'])}",
    ]
    if include_trace:
        lines.append(f"trace_events: {len(data.get('trace', []))}")
    return "\n".join(lines)

def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    divider = "  ".join("-" * widths[col] for col in columns)
    body = [
        "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])

def score_label(score_summary: dict[str, Any]) -> str:
    metrics = score_summary.get("metrics") or {}
    if len(metrics) == 1:
        key, value = next(iter(metrics.items()))
        return f"{key}={value}"
    if metrics:
        return json.dumps(metrics, sort_keys=True)
    return str(score_summary.get("details", ""))

def task_summary(spec: Any) -> dict[str, Any]:
    metadata = spec.metadata or {}
    name = metadata.get("category") or spec.task_id.rsplit("/", 1)[-1]
    return {
        "task_id": spec.task_id,
        "level": metadata.get("level", ""),
        "name": name,
        "default_instance_id": spec.default_instance_id,
    }
