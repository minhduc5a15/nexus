"""Validate the stronger-model control protocol without calling Modal or a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.model_control_protocol import (
    ROOT,
    ProtocolError,
    load_json,
    validate_canary_gate,
    validate_control_protocol,
    validate_development_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evals/model_control_manifest_v1.json",
        help="Tracked control manifest to verify",
    )
    parser.add_argument(
        "--canary-report",
        type=Path,
        help="Optional downloaded canary report to validate against the manifest",
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        help="Optional downloaded development report; requires --canary-report",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = validate_control_protocol(root=ROOT, manifest_path=args.manifest)
        canary_gate = None
        if args.canary_report:
            manifest = load_json(args.manifest)
            report = load_json(args.canary_report)
            run_id = report.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ProtocolError("Canary report has no run ID")
            canary_gate = validate_canary_gate(
                report, manifest, expected_run_id=run_id
            )
            summary["canary_gate"] = canary_gate
        if args.development_report:
            if canary_gate is None:
                raise ProtocolError("--development-report requires --canary-report")
            manifest = load_json(args.manifest)
            report = load_json(args.development_report)
            run_id = report.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ProtocolError("Development report has no run ID")
            summary["development"] = validate_development_report(
                report,
                manifest,
                expected_run_id=run_id,
                expected_canary_gate=canary_gate,
            )
    except ProtocolError as error:
        print(f"Control protocol invalid: {error}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Preflight only: no Modal function, model, GPU, tool or final holdout was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
