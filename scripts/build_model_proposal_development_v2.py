"""Build a fresh, action-balanced proposal development set without model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.eval_proposals import ACTION_TO_TOOL


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals/model_proposal_development_v2.json"


def _call(action: str, index: int, prompt: str, arguments: dict[str, Any], dimension: str) -> dict[str, Any]:
    return {
        "id": f"dev2_{action}_{index:02d}",
        "prompt": prompt,
        "family": f"dev2_{action}_frame_{(index - 1) // 2 + 1}",
        "rationale": "Yêu cầu nêu rõ hành động và đủ tham số; proposal phải giữ đúng giá trị trong lời người dùng.",
        "dimensions": [dimension],
        "origin": "authored_development_v2",
        "expected": {
            "behavior": "call",
            "action": action,
            "accepted_calls": [{"name": ACTION_TO_TOOL[action], "arguments": arguments}],
            "missing_slots": [],
        },
    }


def _control(index: int, behavior: str, action: str | None, prompt: str, slots: list[str], dimension: str) -> dict[str, Any]:
    case = {
        "id": f"dev2_control_{index:02d}",
        "prompt": prompt,
        "family": f"dev2_control_{index:02d}",
        "rationale": "Không được gọi tool khi thiếu dữ liệu, yêu cầu mơ hồ, bị phủ định hoặc nằm ngoài contract.",
        "dimensions": [dimension],
        "origin": "authored_development_v2",
        "expected": {"behavior": behavior, "action": action, "accepted_calls": [], "missing_slots": slots},
    }
    if behavior == "clarify":
        case["known_fields"] = {}
    return case


def build_document() -> dict[str, Any]:
    # Each pair shares a syntactic frame, but payloads and IDs differ. These are
    # newly authored diagnostic families, not perturbations of the three failures.
    rows: dict[str, list[tuple[str, dict[str, Any], str]]] = {
        "create": [
            ("Đưa vào sổ việc đúng dòng này: thay lõi lọc nước", {"content": "thay lõi lọc nước"}, "content_boundary"),
            ("Đưa vào sổ việc đúng dòng này: gọi cô giáo hỏi lịch", {"content": "gọi cô giáo hỏi lịch"}, "content_boundary"),
            ("Tạo một mục mới với nội dung nguyên văn 'đọc RFC 9110'", {"content": "đọc RFC 9110"}, "verbatim"),
            ("Tạo một mục mới với nội dung nguyên văn 'kiểm tra SSD-02'", {"content": "kiểm tra SSD-02"}, "verbatim"),
            ("Ghi các dòng sau thành một lần thêm việc:\nquét sân\nrửa bình nước", {"content": "quét sân\nrửa bình nước"}, "multiline"),
            ("Ghi các dòng sau thành một lần thêm việc:\nchạy test\nxem log lỗi", {"content": "chạy test\nxem log lỗi"}, "multiline"),
            ("Lưu thành task, phần chữ là: phân loại vít M4, M5", {"content": "phân loại vít M4, M5"}, "content_boundary"),
            ("Lưu thành task, phần chữ là: lau bảng & dọn bút", {"content": "lau bảng & dọn bút"}, "content_boundary"),
        ],
        "list": [
            ("Đọc lại tất cả task đang lưu cho tôi.", {}, "all_tasks"),
            ("Đọc lại toàn bộ đầu việc đang lưu cho tôi.", {}, "all_tasks"),
            ("Tôi cần bản kê đầy đủ các mục đã nhập vào NEXUS.", {}, "all_tasks"),
            ("Tôi cần bản kê đầy đủ các việc đã nhập vào NEXUS.", {}, "all_tasks"),
            ("Mở danh mục việc, tính cả những mục đã hoàn thành.", {}, "includes_completed"),
            ("Mở sổ task, tính cả những mục đã hoàn thành.", {}, "includes_completed"),
            ("Liệt kê mọi việc tôi đang giữ trong ứng dụng.", {}, "all_tasks"),
            ("Liệt kê mọi task tôi đang giữ trong ứng dụng.", {}, "all_tasks"),
        ],
        "complete": [
            ("Tôi đã làm xong mục mang ID 318; cập nhật giúp.", {"id": 318}, "id_exact"),
            ("Tôi đã làm xong mục mang ID 547; cập nhật giúp.", {"id": 547}, "id_exact"),
            ("Chuyển trạng thái của việc #729 sang đã hoàn thành.", {"id": 729}, "id_exact"),
            ("Chuyển trạng thái của việc #864 sang đã hoàn thành.", {"id": 864}, "id_exact"),
            ("Tick xong cho đúng task số 1092.", {"id": 1092}, "id_exact"),
            ("Tick xong cho đúng task số 1734.", {"id": 1734}, "id_exact"),
            ("Đánh dấu hoàn tất bản ghi việc 2506 nhé.", {"id": 2506}, "id_exact"),
            ("Đánh dấu hoàn tất bản ghi việc 3871 nhé.", {"id": 3871}, "id_exact"),
        ],
        "edit": [
            ("Sửa chữ của mục #318, thay bằng: thử mạng LAN", {"id": 318, "content": "thử mạng LAN"}, "content_boundary"),
            ("Sửa chữ của mục #547, thay bằng: kiểm kê ốc vít", {"id": 547, "content": "kiểm kê ốc vít"}, "content_boundary"),
            ("Đổi toàn bộ nội dung task 729 sang 'Gửi bản PDF-aB'", {"id": 729, "content": "Gửi bản PDF-aB"}, "verbatim"),
            ("Đổi toàn bộ nội dung task 864 sang 'Hỏi chị Hòa'", {"id": 864, "content": "Hỏi chị Hòa"}, "verbatim"),
            ("Với ID 1092, viết lại phần nội dung thành: đặt giấy A3, loại nhám", {"id": 1092, "content": "đặt giấy A3, loại nhám"}, "content_boundary"),
            ("Với ID 1734, viết lại phần nội dung thành: cập nhật C++: ranges", {"id": 1734, "content": "cập nhật C++: ranges"}, "content_boundary"),
            ("Thay text của việc số 2506 bằng đúng 'đọc chương Hai'", {"id": 2506, "content": "đọc chương Hai"}, "verbatim"),
            ("Thay text của việc số 3871 bằng đúng 'in nhãn SKU-9'", {"id": 3871, "content": "in nhãn SKU-9"}, "verbatim"),
        ],
        "delete": [
            ("Tôi muốn xóa vĩnh viễn mục ID 318; hãy hỏi xác nhận ở ứng dụng.", {"id": 318}, "id_exact"),
            ("Tôi muốn xóa vĩnh viễn mục ID 547; hãy hỏi xác nhận ở ứng dụng.", {"id": 547}, "id_exact"),
            ("Loại task #729 ra khỏi danh sách sau bước xác nhận.", {"id": 729}, "id_exact"),
            ("Loại task #864 ra khỏi danh sách sau bước xác nhận.", {"id": 864}, "id_exact"),
            ("Bỏ hẳn bản ghi công việc số 1092 đi.", {"id": 1092}, "id_exact"),
            ("Bỏ hẳn bản ghi công việc số 1734 đi.", {"id": 1734}, "id_exact"),
            ("Remove riêng việc 2506, không đụng tới việc khác.", {"id": 2506}, "id_exact"),
            ("Remove riêng việc 3871, không đụng tới việc khác.", {"id": 3871}, "id_exact"),
        ],
        "deadline": [
            ("Chốt hạn 6 giờ 20 phút sáng mai cho task #318.", {"id": 318, "when": "6 giờ 20 phút sáng mai"}, "relative_time"),
            ("Chốt hạn 9 giờ 10 phút tối ngày mai cho task #547.", {"id": 547, "when": "9 giờ 10 phút tối ngày mai"}, "relative_time"),
            ("Hãy gắn mốc 16:40 hôm nay vào việc mang ID 729.", {"id": 729, "when": "16:40 hôm nay"}, "relative_time"),
            ("Hãy gắn mốc 07:25 ngày mai vào việc mang ID 864.", {"id": 864, "when": "07:25 ngày mai"}, "relative_time"),
            ("Deadline mới của việc 1092: 10 giờ sáng hôm nay; cập nhật nó.", {"id": 1092, "when": "10 giờ sáng hôm nay"}, "relative_time"),
            ("Deadline mới của việc 1734: 8 giờ tối mai; cập nhật nó.", {"id": 1734, "when": "8 giờ tối mai"}, "relative_time"),
            ("Gán ngày giờ 14/07/2028 09:35 làm hạn cho ID 2506.", {"id": 2506, "when": "14/07/2028 09:35"}, "absolute_time"),
            ("Gán ngày giờ 20:50 ngày 02/08/2028 làm hạn cho ID 3871.", {"id": 3871, "when": "20:50 ngày 02/08/2028"}, "absolute_time"),
        ],
        "due_query": [
            ("Truy vấn riêng nhóm task có deadline thuộc ngày hôm nay.", {"scope": "today"}, "scope_exact"),
            ("Truy vấn riêng nhóm việc có deadline thuộc ngày hôm nay.", {"scope": "today"}, "scope_exact"),
            ("Các mục chưa xong nào chạm hạn vào ngày kế tiếp, tức ngày mai?", {"scope": "tomorrow"}, "scope_exact"),
            ("Các task chưa xong nào chạm hạn vào ngày kế tiếp, tức ngày mai?", {"scope": "tomorrow"}, "scope_exact"),
            ("Mở nhóm công việc đã trễ deadline.", {"scope": "overdue"}, "scope_exact"),
            ("Mở nhóm task đã trễ deadline.", {"scope": "overdue"}, "scope_exact"),
            ("Cho tôi xem việc quá hạn trong NEXUS.", {"scope": "overdue"}, "scope_exact"),
            ("Cho tôi xem task quá hạn trong NEXUS.", {"scope": "overdue"}, "scope_exact"),
        ],
    }
    cases = [_call(action, index, prompt, args, dimension)
             for action, group in rows.items()
             for index, (prompt, args, dimension) in enumerate(group, 1)]
    controls = [
        ("clarify", "create", "Tạo cho tôi một việc; nội dung tôi sẽ nói ở tin tiếp.", ["content"], "missing_content"),
        ("clarify", "complete", "Đánh dấu một việc là xong, tôi chưa chọn số.", ["id"], "missing_id"),
        ("clarify", "edit", "Sửa task 410 giúp, câu thay thế tôi chưa gửi.", ["content"], "missing_content"),
        ("clarify", "edit", "Đổi một mục thành gửi phiếu bảo hành; ID sẽ báo sau.", ["id"], "missing_id"),
        ("clarify", "delete", "Xóa một mục khỏi sổ, chờ tôi nói số ID.", ["id"], "missing_id"),
        ("clarify", "deadline", "Đặt hạn 11:45 ngày mai cho một việc chưa nêu ID.", ["id"], "missing_id"),
        ("clarify", "deadline", "Task #411 cần mốc chót, nhưng giờ tôi chưa quyết.", ["when"], "missing_when"),
        ("clarify", "deadline", "Chọn một trong hai việc 412 và 413 để gắn hạn 14:10 mai.", ["id"], "multiple_ids"),
        ("clarify", "deadline", "Đặt hạn của việc 414 vào 7 giờ hay 8 giờ tối mai, tôi chưa chốt.", ["when"], "multiple_times"),
        ("clarify", "due_query", "Mở danh sách việc theo hạn, để tôi chọn phạm vi sau.", ["scope"], "missing_scope"),
        ("clarify", "due_query", "Tôi muốn xem việc quá hạn hoặc ngày mai, chưa chọn một phạm vi.", ["scope"], "multiple_scopes"),
        ("clarify", None, "Hôm nay có gì quan trọng vậy?", ["intent"], "ambiguous_intent"),
        ("no_action", None, "Câu mẫu trên tờ giấy là 'xóa việc 901'; tôi đang trích dẫn.", [], "quoted_command"),
        ("no_action", None, "Việc 902 cứ để nguyên như cũ; đừng tick là đã xong.", [], "negation"),
        ("no_action", None, "Không cần tạo task rửa xe hôm nay.", [], "negation"),
        ("no_action", None, "Tôi có nhắc 8 giờ sáng mai nhưng chưa muốn gắn hạn cho việc 903.", [], "negation"),
        ("no_action", None, "Tôi vừa nghe một người nói 'xem danh sách', chỉ kể lại thôi.", [], "quoted_command"),
        ("no_action", None, "Mua đường. Đây chỉ là ghi chú trong câu hỏi của tôi.", [], "bare_statement"),
        ("no_action", None, "Tôi muốn bỏ mốc hạn đang gắn trên việc 904.", [], "unsupported_action"),
        ("no_action", None, "Cho tôi xem các việc tới hạn vào ngày 04/06/2028.", [], "unsupported_scope"),
        ("no_action", None, "Nhắc tôi về việc 905 mỗi sáng.", [], "unsupported_action"),
        ("no_action", None, "Hãy tự chọn task gần giống chữ 'mua pin' rồi xóa.", [], "unsupported_reference"),
        ("no_action", None, "Tôi muốn đưa ra ví dụ: 'đặt hạn việc 906 lúc 8 giờ tối mai'.", [], "quoted_command"),
        ("no_action", None, "Không sửa nội dung task 907 dù nó sai chính tả.", [], "negation"),
    ]
    cases.extend(_control(index, *row) for index, row in enumerate(controls, 1))
    return {
        "schema_version": 1,
        "split": "development",
        "version": 2,
        "purpose": "New diagnostic development set for comparing base, step-300 and future candidates; not a blind final holdout.",
        "label_source": "Authored from the product contract; model, policy and classifier outputs were not used as gold. Human review pending.",
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
        parser.error("Tracked development v2 differs from its builder")
    print(f"Development v2: {len(build_document()['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
