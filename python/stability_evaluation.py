#!/usr/bin/env python3
"""Offline evaluation for coach-labelled trunk-stability intervals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


Range = tuple[int, int]


def normalized_ranges(values: Iterable[Iterable[Any]] | None) -> list[Range]:
    ranges: list[Range] = []
    for item in values or []:
        pair = list(item)
        if len(pair) < 2:
            continue
        start = max(0, int(pair[0]))
        end = max(start, int(pair[1]))
        ranges.append((start, end))
    return sorted(ranges)


def time_in_ranges(time_ms: int, ranges: list[Range]) -> bool:
    return any(start <= time_ms < end for start, end in ranges)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def confusion_metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    tn = int(counts.get("tn", 0))
    fn = int(counts.get("fn", 0))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "falsePositiveRate": safe_ratio(fp, fp + tn),
        "accuracy": safe_ratio(tp + tn, tp + fp + tn + fn),
    }


def range_iou(left: Range, right: Range) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union else 0.0


def boundary_metrics(expected: list[Range], predicted: list[Range]) -> dict[str, Any]:
    remaining = set(range(len(predicted)))
    matches: list[dict[str, Any]] = []
    for expected_range in expected:
        candidates = [
            (range_iou(expected_range, predicted[index]), index)
            for index in remaining
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index < 0 or best_iou <= 0.0:
            continue
        remaining.remove(best_index)
        predicted_range = predicted[best_index]
        matches.append({
            "expected": list(expected_range),
            "predicted": list(predicted_range),
            "iou": round(best_iou, 4),
            "onsetErrorMs": abs(predicted_range[0] - expected_range[0]),
            "offsetErrorMs": abs(predicted_range[1] - expected_range[1]),
        })
    return {
        "expectedSegments": len(expected),
        "predictedSegments": len(predicted),
        "matchedSegments": len(matches),
        "missedSegments": max(0, len(expected) - len(matches)),
        "extraSegments": len(remaining),
        "meanIoU": round(sum(item["iou"] for item in matches) / len(matches), 4) if matches else None,
        "meanOnsetErrorMs": round(sum(item["onsetErrorMs"] for item in matches) / len(matches), 1) if matches else None,
        "meanOffsetErrorMs": round(sum(item["offsetErrorMs"] for item in matches) / len(matches), 1) if matches else None,
        "matches": matches,
    }


def cohens_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_positive = sum(left) / len(left)
    right_positive = sum(right) / len(right)
    expected = left_positive * right_positive + (1.0 - left_positive) * (1.0 - right_positive)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return round((observed - expected) / (1.0 - expected), 4)


def unstable_ranges_from_segments(segments: Iterable[dict[str, Any]]) -> list[Range]:
    return normalized_ranges([
        [item.get("startTimeMs"), item.get("endTimeMs")]
        for item in segments
        if item.get("state") == "unstable"
    ])


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    judgments = list(result.get("frameJudgments") or [])
    expected = normalized_ranges(case.get("consensusUnstableRangesMs") or case.get("unstableRangesMs"))
    ignored = normalized_ranges(case.get("ignoreRangesMs"))
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    evaluated_times: list[int] = []
    for item in judgments:
        time_ms = int(item.get("timeMs") or 0)
        if item.get("state") in {"unknown", "not_evaluated"} or time_in_ranges(time_ms, ignored):
            continue
        actual = time_in_ranges(time_ms, expected)
        predicted = item.get("state") == "unstable"
        counts["tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"] += 1
        evaluated_times.append(time_ms)

    predicted_ranges = unstable_ranges_from_segments(result.get("judgmentSegments") or [])
    annotators = list(case.get("annotators") or [])
    kappa = None
    if len(annotators) >= 2 and evaluated_times:
        left_ranges = normalized_ranges(annotators[0].get("unstableRangesMs"))
        right_ranges = normalized_ranges(annotators[1].get("unstableRangesMs"))
        kappa = cohens_kappa(
            [time_in_ranges(time_ms, left_ranges) for time_ms in evaluated_times],
            [time_in_ranges(time_ms, right_ranges) for time_ms in evaluated_times],
        )

    return {
        "id": str(case.get("id") or "unknown"),
        "actionType": str(case.get("actionType") or "unknown"),
        "view": str(case.get("view") or "unknown"),
        "evaluatedFrames": len(evaluated_times),
        "ignoredOrUnknownFrames": max(0, len(judgments) - len(evaluated_times)),
        "confusion": confusion_metrics(counts),
        "boundaries": boundary_metrics(expected, predicted_ranges),
        "coachAgreementKappa": kappa,
    }


def analysis_video_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "frameJudgments" in payload:
        return payload
    videos = (payload.get("analysis") or {}).get("videos") or payload.get("videos") or []
    return videos[0] if videos else {}


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[report["actionType"]].append(report)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {key: sum(int(item["confusion"][key]) for item in items) for key in ("tp", "fp", "tn", "fn")}
        kappas = [float(item["coachAgreementKappa"]) for item in items if item.get("coachAgreementKappa") is not None]
        onset = [float(item["boundaries"]["meanOnsetErrorMs"]) for item in items if item["boundaries"].get("meanOnsetErrorMs") is not None]
        offset = [float(item["boundaries"]["meanOffsetErrorMs"]) for item in items if item["boundaries"].get("meanOffsetErrorMs") is not None]
        return {
            "videos": len(items),
            "confusion": confusion_metrics(counts),
            "meanCoachAgreementKappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
            "meanOnsetErrorMs": round(sum(onset) / len(onset), 1) if onset else None,
            "meanOffsetErrorMs": round(sum(offset) / len(offset), 1) if offset else None,
        }

    return {
        "overall": summarize(reports),
        "byAction": {action: summarize(items) for action, items in sorted(grouped.items())},
        "cases": reports,
    }


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, dict) else manifest
    reports: list[dict[str, Any]] = []
    for case in cases or []:
        result_path = (manifest_path.parent / str(case["resultPath"])).resolve()
        result = analysis_video_payload(json.loads(result_path.read_text(encoding="utf-8")))
        reports.append(evaluate_case(case, result))
    return aggregate_reports(reports)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate stability judgments against coach-labelled intervals.")
    parser.add_argument("manifest", type=Path, help="UTF-8 JSON manifest with cases and resultPath entries")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest.resolve())
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
