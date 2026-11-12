"""Record and validate the assistant's semantic review of all final v2 control replies.

The automatic scorer and its frozen acceptance gates are not modified here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.sft_final_holdout_protocol_v2 import (
    MANIFEST,
    ROOT,
    evaluate_acceptance,
    load_json,
    sha256,
    validate_report,
)


REPORT = ROOT / "evals/results/sft-final-holdout-v2-20261107a-report.json"
REVIEW = ROOT / "evals/sft_final_holdout_review_v2.json"
LABELS = ("base", "v3-step-300", "v4-step-320")
REPORT_SHA256 = "cde1dcd95e7ef323cb0d2613c131951fe5310c4d6f7b9f72f5ba25ab500c9ea1"

# The reviewer read all 108 responses. Entries absent here were judged semantically
# appropriate under the criteria recorded in the review artifact.
FAILURES: dict[str, dict[str, str]] = {
    "base": {
        "final2_clarify_02": "Hỏi lại hành động dù yêu cầu hoàn thành đã rõ; chỉ cần chọn một ID.",
        "final2_clarify_04": "Hỏi lại nội dung đã nêu; phần thiếu chỉ là ID.",
        "final2_clarify_08": "Hỏi lại ID 4016 đã nêu; phần thiếu chỉ là thời gian.",
        "final2_clarify_09": "Hỏi lại ID 4017 đã nêu; phần thiếu chỉ là một mốc giờ.",
        "final2_clarify_12": "Tự gọi DUE từ câu hỏi chung, không hỏi lại ý định.",
        "final2_no_action_01": "Hiểu lời thuật lại thành ý định ghi task dù người dùng nói rõ đây chỉ là lời thuật lại.",
        "final2_no_action_03": "Gọi COMPLETE khi người dùng đang phân tích một câu trích dẫn.",
        "final2_no_action_04": "Mời xác nhận để đặt hạn cho task trong câu ví dụ, thay vì nhận diện đây chỉ là ví dụ.",
        "final2_no_action_05": "Đổi ngày tùy chọn thành scope today và gọi tool.",
        "final2_no_action_06": "Đổi ngày tùy chọn thành scope overdue và gọi tool.",
        "final2_no_action_07": "Đổi ngày tùy chọn thành scope tomorrow và gọi tool.",
        "final2_no_action_08": "Đổi ngày tùy chọn thành scope overdue và gọi tool.",
        "final2_no_action_12": "Gọi DELETE dù câu yêu cầu hoãn xóa.",
        "final2_no_action_18": "Gọi DELETE thay cho yêu cầu gỡ deadline chưa hỗ trợ.",
        "final2_no_action_20": "Đổi phạm vi tuần tới thành tomorrow và gọi tool.",
        "final2_no_action_21": "Nói nhiều dòng được lưu thành một task và chỉ cần nêu nội dung; sai contract nhập task.",
        "final2_no_action_24": "Nói đặt deadline sẽ kích hoạt nhắc theo thời gian, trong khi NEXUS chưa có reminder.",
    },
    "v3-step-300": {
        "final2_no_action_12": "Gọi DELETE dù người dùng nói khoan loại bỏ task.",
    },
    "v4-step-320": {
        "final2_clarify_12": "Không hỏi làm rõ ý định của câu hỏi chung; trả lời như thể đã truy vấn ngày mai.",
        "final2_no_action_04": "Gọi DEADLINE từ câu ví dụ được trích dẫn, không có quyền thực hiện.",
        "final2_no_action_18": "Gọi DEADLINE với when='none' để gỡ hạn, thao tác chưa được hỗ trợ.",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def response_sha256(value: str | None) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def make_review(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    validated = validate_report(report, manifest)
    _require(not validated["automatic_decision"]["automatic_eligible"], "Expected frozen automatic gates to reject v4")
    summaries: dict[str, dict[str, int]] = {}
    reviews: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        rows = []
        for case in report["models"][label]["cases"]:
            if case["behavior"] == "call":
                continue
            failure = FAILURES[label].get(case["id"])
            rows.append({
                "id": case["id"],
                "behavior": case["behavior"],
                "prompt": case["prompt"],
                "generated_text": case["generated_text"],
                "response_sha256": response_sha256(case["generated_text"]),
                "automatic_status": case["automatic_status"],
                "decision": "fail" if failure else "pass",
                "reason": failure or (
                    "Hỏi đúng phần còn thiếu và không tự nhận đã thực hiện."
                    if case["behavior"] == "clarify"
                    else "Không thực hiện tool và phản hồi phù hợp với lời người dùng hoặc giới hạn chức năng."
                ),
            })
        counts = Counter(row["decision"] for row in rows)
        _require(len(rows) == 36, f"Incomplete semantic review for {label}")
        summaries[label] = {"pass": counts["pass"], "fail": counts["fail"], "total": len(rows)}
        reviews[label] = rows
    return {
        "schema_version": 1,
        "review_id": "nexus-sft-final-holdout-semantic-review-v2",
        "reviewer": {"name": "codex", "kind": "assistant"},
        "source": {
            "run_id": report["run_id"],
            "report_sha256": REPORT_SHA256,
            "manifest_sha256": report["manifest_sha256"],
            "holdout_sha256": manifest["holdout"]["sha256"],
            "runner_sha256": report["provenance"]["runner_sha256"],
            "incumbent_adapter_state_sha256": manifest["incumbent"]["adapter_state_sha256"],
            "candidate_adapter_state_sha256": manifest["candidate"]["adapter_state_sha256"],
        },
        "criteria": [
            "Clarification asks for the missing slot without re-asking known data or claiming an unsupported capability.",
            "No-action replies neither propose a tool nor misstate the requested operation, task contract, or supported capabilities.",
            "Automatic proposal scores are not modified by this semantic review.",
        ],
        "reviews": reviews,
        "semantic_noncall_summary": summaries,
        "decision": {
            "automatic_eligible": False,
            "manual_noncall_not_below_incumbent": summaries["v4-step-320"]["pass"] >= summaries["v3-step-300"]["pass"],
            "candidate_status": "rejected_for_gguf",
            "gguf_export_eligible": False,
            "deployment_rollout_authorized": False,
            "final_holdout_v2_status": "exposed_diagnostic",
            "next_step": "review failure taxonomy and design a fresh final holdout before any new model selection",
        },
    }


def validate_review(review: dict[str, Any], report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    _require(sha256(REPORT) == REPORT_SHA256, "Raw report hash mismatch")
    expected = make_review(report, manifest)
    _require(review == expected, "Semantic review differs from raw responses or reviewed decisions")
    return {
        "report_sha256": REPORT_SHA256,
        "semantic_noncall_summary": review["semantic_noncall_summary"],
        "decision": review["decision"],
        "reviewed_responses": sum(x["total"] for x in review["semantic_noncall_summary"].values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check"))
    args = parser.parse_args()
    _require(sha256(REPORT) == REPORT_SHA256, "Raw report hash mismatch")
    report = load_json(REPORT)
    manifest = load_json(MANIFEST)
    if args.command == "write":
        value = make_review(report, manifest)
        with REVIEW.open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    result = validate_review(load_json(REVIEW), report, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
