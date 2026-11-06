"""Build a small, balanced development set for the reviewed SFT v4 candidate.

Gold labels come from the product contract, never from model or policy output.
This split is diagnostic and must not be used as training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.eval_proposals import ACTION_TO_TOOL


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals/model_proposal_development_v3.json"


def _call(action: str, index: int, prompt: str, arguments: dict[str, Any], dimension: str) -> dict:
    return {
        "id": f"dev3_{action}_{index:02d}",
        "prompt": prompt,
        "family": f"dev3_{action}_frame_{(index - 1) // 2 + 1}",
        "rationale": "Yêu cầu nêu rõ một hành động và đủ tham số; proposal phải giữ nguyên ID và dữ liệu người dùng nêu.",
        "dimensions": [dimension],
        "origin": "authored_development_v3",
        "expected": {
            "behavior": "call",
            "action": action,
            "accepted_calls": [{"name": ACTION_TO_TOOL[action], "arguments": arguments}],
            "missing_slots": [],
        },
    }


def _control(index: int, behavior: str, action: str | None, prompt: str, slots: list[str], dimension: str) -> dict:
    case = {
        "id": f"dev3_control_{index:02d}",
        "prompt": prompt,
        "family": f"dev3_control_{index:02d}",
        "rationale": "Không thực thi khi yêu cầu thiếu tham số, bị phủ định, chỉ là trích dẫn hoặc nằm ngoài contract.",
        "dimensions": [dimension],
        "origin": "authored_development_v3",
        "expected": {"behavior": behavior, "action": action, "accepted_calls": [], "missing_slots": slots},
    }
    if behavior == "clarify":
        case["known_fields"] = {}
    return case


def build_document() -> dict:
    # Each adjacent pair shares a frame within this split. Frames are authored
    # independently of the 40 new training examples and the older diagnostics.
    rows: dict[str, list[tuple[str, dict[str, Any], str]]] = {
        "create": [
            ("Cho vào danh sách một việc mới: kiểm tra van nước sân sau", {"content": "kiểm tra van nước sân sau"}, "content_boundary"),
            ("Cho vào danh sách một việc mới: gửi ảnh bảng mạch cho Dũng", {"content": "gửi ảnh bảng mạch cho Dũng"}, "content_boundary"),
            ("Tôi muốn lưu nguyên dòng sau làm task:\nđọc mục 4.2 và ghi câu hỏi", {"content": "đọc mục 4.2 và ghi câu hỏi"}, "verbatim"),
            ("Tôi muốn lưu nguyên dòng sau làm task:\nkiểm tra cổng USB, dây HDMI", {"content": "kiểm tra cổng USB, dây HDMI"}, "verbatim"),
        ],
        "list": [
            ("Danh sách việc đang lưu trong NEXUS gồm những mục nào?", {}, "all_tasks"),
            ("Danh sách task đang lưu trong NEXUS gồm những mục nào?", {}, "all_tasks"),
            ("Mở sổ cho tôi kiểm tra tất cả việc, dù đã xong hay chưa.", {}, "includes_completed"),
            ("Mở sổ cho tôi kiểm tra tất cả task, dù đã xong hay chưa.", {}, "includes_completed"),
        ],
        "complete": [
            ("Mục số 184 vừa xử lý xong; đánh dấu hoàn tất giúp tôi.", {"id": 184}, "id_exact"),
            ("Mục số 296 vừa xử lý xong; đánh dấu hoàn tất giúp tôi.", {"id": 296}, "id_exact"),
            ("Cập nhật trạng thái việc 743 sang đã làm xong.", {"id": 743}, "id_exact"),
            ("Cập nhật trạng thái việc 851 sang đã làm xong.", {"id": 851}, "id_exact"),
        ],
        "edit": [
            ("Tôi gõ sai ở mục 572; hãy sửa mục ấy, ghi nội dung mới là 'xem lại sơ đồ điện'.", {"id": 572, "content": "xem lại sơ đồ điện"}, "content_boundary"),
            ("Tôi gõ sai ở mục 683; hãy sửa mục ấy, ghi nội dung mới là 'hỏi giá bộ cảm biến'.", {"id": 683, "content": "hỏi giá bộ cảm biến"}, "content_boundary"),
            ("Ở việc 194, thay phần chữ hiện tại bằng 'mua đầu nối RJ45'", {"id": 194, "content": "mua đầu nối RJ45"}, "verbatim"),
            ("Ở việc 285, thay phần chữ hiện tại bằng 'đọc chương về async'", {"id": 285, "content": "đọc chương về async"}, "verbatim"),
        ],
        "delete": [
            ("Xóa hẳn mục mang số 903, hãy cho tôi xác nhận trước.", {"id": 903}, "id_exact"),
            ("Xóa hẳn mục mang số 914, hãy cho tôi xác nhận trước.", {"id": 914}, "id_exact"),
            ("Tôi muốn loại việc 365 khỏi NEXUS sau khi xác nhận.", {"id": 365}, "id_exact"),
            ("Tôi muốn loại việc 476 khỏi NEXUS sau khi xác nhận.", {"id": 476}, "id_exact"),
        ],
        "deadline": [
            ("Hạn chót của việc 417 là 13:05 ngày mai; lưu mốc đó.", {"id": 417, "when": "13:05 ngày mai"}, "relative_time"),
            ("Hạn chót của việc 528 là 18:25 hôm nay; lưu mốc đó.", {"id": 528, "when": "18:25 hôm nay"}, "relative_time"),
            ("Với mục 739, hãy ghi deadline 8 giờ 15 phút sáng mai.", {"id": 739, "when": "8 giờ 15 phút sáng mai"}, "relative_time"),
            ("Với mục 841, hãy ghi deadline 9 giờ 40 phút tối ngày mai.", {"id": 841, "when": "9 giờ 40 phút tối ngày mai"}, "relative_time"),
        ],
        "due_query": [
            ("Những task chưa hoàn thành nào có deadline trong ngày mai?", {"scope": "tomorrow"}, "scope_exact"),
            ("Những việc chưa hoàn thành nào có deadline trong ngày mai?", {"scope": "tomorrow"}, "scope_exact"),
            ("Lấy ra các mục đã qua hạn tính tới hiện tại.", {"scope": "overdue"}, "scope_exact"),
            ("Lấy ra các task đã qua hạn tính tới hiện tại.", {"scope": "overdue"}, "scope_exact"),
        ],
    }
    cases = [
        _call(action, index, prompt, arguments, dimension)
        for action, group in rows.items()
        for index, (prompt, arguments, dimension) in enumerate(group, 1)
    ]
    controls = [
        ("clarify", "create", "Tôi cần lưu một task nữa, phần nội dung sẽ gửi ở tin sau.", ["content"], "missing_content"),
        ("clarify", "complete", "Tôi làm xong một mục rồi; chưa nhớ số ID của nó.", ["id"], "missing_id"),
        ("clarify", "edit", "Đổi nội dung việc 534 giúp tôi, để tôi gửi câu mới sau.", ["content"], "missing_content"),
        ("clarify", "delete", "Tôi muốn xóa một task, nhưng chưa chọn ID.", ["id"], "missing_id"),
        ("clarify", "deadline", "Đặt hạn việc 645, thời gian tôi sẽ báo tiếp.", ["when"], "missing_when"),
        ("clarify", "due_query", "Xem các mục theo hạn; tôi chưa chọn hôm nay, mai hay quá hạn.", ["scope"], "missing_scope"),
        ("no_action", None, "Trong tài liệu có câu 'xóa việc 956'; tôi chỉ đang đọc ví dụ.", [], "quoted_command"),
        ("no_action", None, "Đừng đổi nội dung việc 167 dù câu cũ hơi khó hiểu.", [], "negation"),
        ("no_action", None, "Không cần ghi thêm task mua ổ cắm lúc này.", [], "negation"),
        ("no_action", None, "Hãy liệt kê riêng các task đã hoàn thành, bỏ hết mục chưa xong.", [], "unsupported_filter"),
        ("no_action", None, "Nhắc lại việc 278 cho tôi vào mỗi thứ hai.", [], "unsupported_reminder"),
        ("no_action", None, "Tôi kể rằng đã đọc xong sách; đây không phải yêu cầu cập nhật task.", [], "bare_statement"),
    ]
    cases.extend(_control(index, *row) for index, row in enumerate(controls, 1))
    return {
        "schema_version": 1,
        "split": "development",
        "version": 3,
        "purpose": "Diagnostic comparison of base, step-300 and reviewed SFT v4 candidate; not a final holdout.",
        "label_source": "Authored from the product contract without model, classifier or policy outputs; owner review pending.",
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
        parser.error("Development v3 differs from builder")
    print(f"Development v3: {len(build_document()['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
