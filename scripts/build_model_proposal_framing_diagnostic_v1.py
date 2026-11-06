"""Build paired DEADLINE/LIST prompts for an exposed, model-only diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals/model_proposal_framing_diagnostic_v1.json"


def _case(pair: str, variant: str, action: str, prompt: str, arguments: dict[str, Any], contrast: str, origin: str) -> dict[str, Any]:
    return {
        "id": f"framing_{pair}_{variant}",
        "pair_id": pair,
        "variant": variant,
        "prompt": prompt,
        "family": f"framing_{pair}_{variant}",
        "rationale": contrast,
        "dimensions": ["paired_framing", "exposed_diagnostic"],
        "origin": origin,
        "expected": {
            "behavior": "call",
            "action": action,
            "accepted_calls": [{
                "name": "set_task_deadline" if action == "deadline" else "list_tasks",
                "arguments": arguments,
            }],
            "missing_slots": [],
        },
    }


def build_document() -> dict[str, Any]:
    specs = [
        ("deadline_01", "deadline", {"id": 6, "when": "8 giờ 30 phút tối ngày mai"},
         "Đặt hạn việc 6 lúc 8 giờ 30 phút tối ngày mai.",
         "Đặt hạn cho việc 6 lúc 8 giờ 30 phút tối ngày mai.",
         "Thêm 'cho' vào khung câu từng bị step-300 bỏ sót.", "old_development_v1"),
        ("deadline_02", "deadline", {"id": 41, "when": "7 giờ sáng hôm nay"},
         "Đặt hạn việc 41 lúc 7 giờ sáng hôm nay.",
         "Chốt hạn 7 giờ sáng hôm nay cho task #41.",
         "Giữ ID/thời gian, đưa thời gian lên trước ID.", "old_development_v1"),
        ("deadline_03", "deadline", {"id": 52, "when": "18:15 ngày mai"},
         "Đặt hạn việc 52 lúc 18:15 ngày mai.",
         "Đặt thời hạn cho việc 52 vào 18:15 ngày mai.",
         "Giữ ID/thời gian, thay khung 'đặt hạn việc ... lúc' bằng khung rõ lời dẫn.", "old_development_v1"),
        ("deadline_04", "deadline", {"id": 1103, "when": "6 giờ 35 phút sáng mai"},
         "Đặt hạn việc 1103 lúc 6 giờ 35 phút sáng mai.",
         "Đặt hạn cho việc 1103 lúc 6 giờ 35 phút sáng mai.",
         "ID/thời gian mới; chỉ thêm 'cho' vào khung câu.", "new_diagnostic"),
        ("deadline_05", "deadline", {"id": 1104, "when": "21:40 hôm nay"},
         "Đặt hạn việc 1104 lúc 21:40 hôm nay.",
         "Chốt hạn 21:40 hôm nay cho task #1104.",
         "ID/thời gian mới; so khung ID-trước với thời-gian-trước.", "new_diagnostic"),
        ("deadline_06", "deadline", {"id": 1105, "when": "9 giờ 20 phút tối ngày mai"},
         "Đặt hạn việc 1105 lúc 9 giờ 20 phút tối ngày mai.",
         "Set deadline task #1105 thành 9 giờ 20 phút tối ngày mai.",
         "ID/thời gian mới; so khung tiếng Việt với khung pha tiếng Anh.", "new_diagnostic"),
        ("list_01", "list", {},
         "Mở danh mục việc, tính cả những mục đã hoàn thành.",
         "Liệt kê mọi việc đã lưu, gồm cả mục đã hoàn thành.",
         "Yêu cầu LIST bao gồm completed; câu đầu là lỗi v2 của step-300.", "old_development_v2"),
        ("list_02", "list", {},
         "Mở sổ task, tính cả những mục đã hoàn thành.",
         "Đọc lại toàn bộ task, kể cả task đã xong.",
         "Yêu cầu LIST bao gồm completed; câu đầu là lỗi v2 của step-300.", "old_development_v2"),
        ("list_03", "list", {},
         "Cho tôi xem sổ việc, cả các mục đã xong.",
         "Tất cả việc trong danh sách, bao gồm đã hoàn thành, hiện là gì?",
         "Hai cách mới để yêu cầu toàn bộ danh sách gồm completed.", "new_diagnostic"),
        ("list_04", "list", {},
         "Mở bảng task, đừng bỏ các task completed.",
         "Xem toàn bộ danh sách task; giữ cả những mục completed.",
         "Hai cách mới pha tiếng Anh để yêu cầu toàn bộ danh sách.", "new_diagnostic"),
    ]
    cases = []
    for pair, action, args, left, right, contrast, origin in specs:
        cases.append(_case(pair, "a", action, left, args, contrast, origin))
        cases.append(_case(pair, "b", action, right, args, contrast, origin))
    return {
        "schema_version": 1,
        "split": "diagnostic",
        "version": 1,
        "purpose": "Paired framing check after development v1/v2 exposure; not an independent holdout or training set.",
        "label_source": "Authored from product contract; three DEADLINE and two LIST prompts intentionally reuse exposed failures.",
        "cases": cases,
    }


def render_document() -> str:
    return json.dumps(build_document(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        OUTPUT.write_text(render_document(), encoding="utf-8")
    elif OUTPUT.read_text(encoding="utf-8") != render_document():
        parser.error("Framing diagnostic differs from builder")
    print(f"Framing diagnostic: {len(build_document()['cases'])} cases / 10 pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
