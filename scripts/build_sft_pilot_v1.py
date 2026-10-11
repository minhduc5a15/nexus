"""Build the reviewed 100-case SFT pilot dataset deterministically."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SANITY_DATASET = ROOT / "evals/sft_sanity_v1.json"
DEFAULT_OUTPUT = ROOT / "evals/sft_pilot_v1.json"


def _tool_call(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def _call(
    suffix: str,
    action: str,
    prompt: str,
    tool: str,
    arguments: dict[str, Any],
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": f"pilot_{suffix}",
        "family": f"pilot_family_{suffix}",
        "behavior": "call",
        "action": action,
        "rationale": rationale,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": _tool_call(tool, arguments)},
        ],
    }


def _reply(
    suffix: str,
    behavior: str,
    action: str | None,
    prompt: str,
    response: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": f"pilot_{suffix}",
        "family": f"pilot_family_{suffix}",
        "behavior": behavior,
        "action": action,
        "rationale": rationale,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def _new_call_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    create = [
        ("create_roast", "Thêm vào danh sách đúng nội dung sau:\nMua cà phê Arabica, loại rang vừa.", "Mua cà phê Arabica, loại rang vừa."),
        ("create_colloquial", "Note giúp tui việc: gọi bác sĩ lúc 14:00", "gọi bác sĩ lúc 14:00"),
        ("create_two_lines", "Ghi lại hai dòng này thành các việc:\nđặt vé tàu\nnhận áo mưa", "đặt vé tàu\nnhận áo mưa"),
        ("create_mixed_language", "Add task này giúp mình: kiểm tra bản build Android", "kiểm tra bản build Android"),
        ("create_action_word", "Nội dung cần đưa nguyên xi vào sổ việc nằm giữa ngoặc vuông: [xóa file nháp]", "xóa file nháp"),
        ("create_colon_content", "Lưu nguyên văn mục này nhé: Ôn C++: vector, map & set.", "Ôn C++: vector, map & set."),
    ]
    for suffix, prompt, content in create:
        cases.append(_call(suffix, "create", prompt, "create_task", {"content": content}, "Sao chép nguyên văn toàn bộ payload CREATE và chỉ gọi một tool."))

    list_prompts = [
        ("list_open_notebook", "Mở sổ task hiện tại cho tôi xem."),
        ("list_colloquial", "Cho mình coi hết mấy việc đã ghi với."),
        ("list_inventory", "Hiện giờ danh sách công việc đang chứa những mục nào?"),
        ("list_mixed_language", "Please show toàn bộ task tôi đã lưu."),
        ("list_enumerate", "Liệt kê tất cả đầu việc trong NEXUS."),
        ("list_review", "Tôi cần xem lại trọn bộ các việc đang có."),
        ("list_question", "Trong sổ việc của tôi hiện có gì vậy?"),
    ]
    for suffix, prompt in list_prompts:
        cases.append(_call(suffix, "list", prompt, "list_tasks", {}, "Yêu cầu xem toàn bộ task không cần arguments."))

    complete = [
        ("complete_plain", "Việc mang ID 42 vừa làm xong; cập nhật trạng thái cho nó.", 42),
        ("complete_hash", "Tôi vừa xử lý xong mục có mã #73; hãy ghi nhận trạng thái này.", 73),
        ("complete_colloquial", "Việc số 106 làm xong rồi, chốt hoàn thành giúp mình.", 106),
        ("complete_mixed_language", "Mark task 219 as complete.", 219),
        ("complete_status", "Chuyển đầu việc 387 sang trạng thái hoàn tất.", 387),
        ("complete_polite", "Phiền bạn ghi nhận việc 504 đã hoàn thành.", 504),
        ("complete_done", "Task mang ID 688 đã làm xong, đánh dấu nó giúp tôi.", 688),
    ]
    for suffix, prompt, task_id in complete:
        cases.append(_call(suffix, "complete", prompt, "complete_task", {"id": task_id}, "Dùng đúng một ID do người dùng nêu để hoàn thành task."))

    edit = [
        ("edit_plain", "Nội dung thay thế của mục ID 51 là: mua pin AA", 51, "mua pin AA"),
        ("edit_hash", "Đổi nội dung task #84 thành Gửi email cho An.", 84, "Gửi email cho An."),
        ("edit_colloquial", "Việc 117 ghi lại thành gọi thợ sửa ống nước nhé", 117, "gọi thợ sửa ống nước"),
        ("edit_mixed_language", "Update task 263 thành kiểm tra pull request #91", 263, "kiểm tra pull request #91"),
        ("edit_punctuation", "Chỉnh đầu việc 399 thành Mua táo, lê & nho.", 399, "Mua táo, lê & nho."),
        ("edit_colon", "Thay nội dung việc 527 bằng Ôn Python: asyncio", 527, "Ôn Python: asyncio"),
        ("edit_multiline", "Đổi task 701 thành đúng hai dòng sau:\nsoát hợp đồng\ngửi kế toán", 701, "soát hợp đồng\ngửi kế toán"),
    ]
    for suffix, prompt, task_id, content in edit:
        cases.append(_call(suffix, "edit", prompt, "update_task", {"id": task_id, "content": content}, "Giữ nguyên ID và payload nội dung mới khi EDIT."))

    delete = [
        ("delete_plain", "Mục mang ID 61 không còn cần nữa; loại vĩnh viễn nó khỏi sổ.", 61),
        ("delete_hash", "Xoá task #95 khỏi danh sách.", 95),
        ("delete_remove", "Bỏ đầu việc 128 ra khỏi sổ task giúp tôi.", 128),
        ("delete_permanent", "Loại vĩnh viễn mục số 274.", 274),
        ("delete_mixed_language", "Delete task 406 giúp mình.", 406),
        ("delete_record", "Tôi muốn xóa bản ghi công việc 538.", 538),
        ("delete_polite", "Phiền bạn loại task mang ID 719 khỏi danh sách.", 719),
    ]
    for suffix, prompt, task_id in delete:
        cases.append(_call(suffix, "delete", prompt, "delete_task", {"id": task_id}, "Chỉ đề xuất DELETE cho đúng ID; ứng dụng xác nhận trước side effect."))

    deadline = [
        ("deadline_tomorrow", "Mốc 8 giờ sáng mai là hạn mới của mục ID 66; gán giúp tôi.", 66, "8 giờ sáng mai"),
        ("deadline_absolute", "Task #109 có deadline 27/09/2027 08:00.", 109, "27/09/2027 08:00"),
        ("deadline_day_order", "Hạn mới dành cho task số 143 là 18:30 ngày 04/11/2027; ghi nhận giúp.", 143, "18:30 ngày 04/11/2027"),
        ("deadline_today", "Gán task 286 mốc 21:15 hôm nay.", 286, "21:15 hôm nay"),
        ("deadline_minutes", "Việc 418 cần hạn 7 giờ 45 phút sáng ngày mai.", 418, "7 giờ 45 phút sáng ngày mai"),
        ("deadline_mixed_language", "For task 549, gán mốc 09:05 mai làm deadline.", 549, "09:05 mai"),
        ("deadline_past_allowed", "Đặt hạn mục 733 thành 01/01/2025 00:30.", 733, "01/01/2025 00:30"),
    ]
    for suffix, prompt, task_id, when in deadline:
        cases.append(_call(suffix, "deadline", prompt, "set_task_deadline", {"id": task_id, "when": when}, "Giữ nguyên ID và cụm thời gian; Python chịu trách nhiệm parse thời gian."))

    due = [
        ("due_today_plain", "Lọc sổ theo phạm vi hôm nay và chỉ lấy mục có deadline.", "today"),
        ("due_today_question", "Hôm nay có task nào tới hạn không?", "today"),
        ("due_tomorrow_plain", "Cho tôi danh sách việc đến hạn ngày mai.", "tomorrow"),
        ("due_tomorrow_colloquial", "Mai mình phải xử lý những task nào đến hạn?", "tomorrow"),
        ("due_overdue_plain", "Liệt kê các đầu việc quá hạn.", "overdue"),
        ("due_overdue_synonym", "Có mục nào đang trễ hạn thì cho mình xem.", "overdue"),
        ("due_mixed_language", "Show me task đến hạn ngày mai.", "tomorrow"),
    ]
    for suffix, prompt, scope in due:
        cases.append(_call(suffix, "due_query", prompt, "list_tasks_by_deadline", {"scope": scope}, "Truy vấn chỉ đọc đúng phạm vi thời gian được nêu."))
    return cases


def _clarification_cases() -> list[dict[str, Any]]:
    specs = [
        ("clarify_create_empty", "create", "Thêm một việc mới giúp tôi.", "Bạn muốn thêm nội dung việc gì?", "CREATE thiếu content."),
        ("clarify_create_whitespace", "create", "Ghi vào danh sách một task, nội dung để tôi nói sau.", "Bạn muốn ghi nội dung nào vào task?", "CREATE chủ động để trống content."),
        ("clarify_complete_missing_id", "complete", "Đánh dấu một việc là hoàn thành giúp mình.", "Bạn muốn hoàn thành việc có ID nào?", "COMPLETE thiếu ID."),
        ("clarify_complete_two_ids", "complete", "Hoàn thành task 32 hoặc 33 giúp tôi.", "Bạn muốn hoàn thành task 32 hay 33? Hãy chọn đúng một ID.", "COMPLETE có nhiều ID."),
        ("clarify_complete_range", "complete", "Đánh dấu việc 70 và 71 là đã xong.", "Bạn muốn hoàn thành việc 70 hay 71? Hãy chọn đúng một ID.", "COMPLETE chỉ hỗ trợ một ID mỗi lượt."),
        ("clarify_edit_missing_id", "edit", "Sửa một task thành mua giấy in.", "Bạn muốn sửa task có ID nào?", "EDIT thiếu ID."),
        ("clarify_edit_missing_content", "edit", "Tôi muốn đổi nội dung việc 92.", "Bạn muốn đổi nội dung việc 92 thành gì?", "EDIT thiếu content mới."),
        ("clarify_edit_two_ids", "edit", "Đổi cả task 135 và 136 thành gọi khách hàng.", "Bạn muốn sửa task 135 hay 136? Hãy chọn đúng một ID.", "EDIT có nhiều ID."),
        ("clarify_edit_ambiguous_payload", "edit", "Chỉnh việc 247 giúp tôi, nhưng nội dung mới chưa chốt.", "Bạn muốn đổi nội dung việc 247 thành gì?", "EDIT chưa có payload đã chốt."),
        ("clarify_edit_id_and_choices", "edit", "Sửa task 378 thành mua trà hoặc mua cà phê.", "Bạn muốn nội dung mới của task 378 là “mua trà” hay “mua cà phê”?", "EDIT có hai payload thay thế không thể tự chọn."),
        ("clarify_delete_missing_id", "delete", "Xóa một task khỏi danh sách.", "Bạn muốn xóa task có ID nào?", "DELETE thiếu ID."),
        ("clarify_delete_two_ids", "delete", "Trong hai mục 462 với 463, tôi cần loại một mục nhưng chưa chọn.", "Bạn muốn xóa việc 462 hay 463? Hãy chọn đúng một ID.", "DELETE có nhiều ID."),
        ("clarify_delete_choice", "delete", "Bỏ task 590 hoặc 591 khỏi sổ.", "Bạn muốn xóa task 590 hay 591? Hãy chọn đúng một ID.", "DELETE không tự chọn giữa hai ID."),
        ("clarify_deadline_missing_id", "deadline", "Đặt deadline 8 giờ sáng mai cho một việc.", "Bạn muốn đặt hạn cho task có ID nào?", "DEADLINE thiếu ID."),
        ("clarify_deadline_missing_time", "deadline", "Đặt hạn mới cho task 624.", "Bạn muốn đặt hạn cho task 624 vào thời gian nào?", "DEADLINE thiếu when."),
        ("clarify_deadline_two_ids", "deadline", "Đặt hạn task 650 và 651 vào 08:00 ngày mai.", "Bạn muốn đặt hạn cho task 650 hay 651? Hãy chọn đúng một ID.", "DEADLINE có nhiều ID."),
        ("clarify_deadline_two_times", "deadline", "Đặt hạn việc 777 vào 8 giờ hoặc 9 giờ sáng mai.", "Bạn muốn đặt hạn việc 777 vào 8 giờ hay 9 giờ sáng mai?", "DEADLINE có hai thời điểm không thể tự chọn."),
        ("clarify_due_missing_scope", "due_query", "Lọc các việc theo hạn cho mình xem.", "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay đã quá hạn?", "DUE query thiếu scope."),
    ]
    return [_reply(suffix, "clarify", action, prompt, response, rationale) for suffix, action, prompt, response, rationale in specs]


def _no_action_cases() -> list[dict[str, Any]]:
    specs = [
        ("no_action_bare_weather", "Trời mưa nên nhớ mang ô.", "Bạn có muốn tôi lưu “Trời mưa nên nhớ mang ô.” thành một task không?", "Câu trần thuật không tự cấp quyền CREATE."),
        ("no_action_bare_groceries", "Tối nay mua thêm trứng.", "Bạn có muốn tôi lưu “Tối nay mua thêm trứng.” thành một task không?", "Câu trần thuật không tự cấp quyền CREATE."),
        ("no_action_bare_question", "Không biết mai có nên gọi cho Nam không nhỉ?", "Khi nào bạn muốn lưu hoặc thao tác task, hãy nói rõ yêu cầu.", "Suy nghĩ thành tiếng không phải yêu cầu thao tác."),
        ("no_action_bare_note", "Bản báo cáo vẫn còn thiếu biểu đồ.", "Bạn có muốn tôi lưu câu này thành một task không?", "Thông tin mô tả không tự tạo task."),
        ("no_action_negated_create", "Đừng thêm việc mua bánh vào danh sách.", "Tôi sẽ không thêm việc đó.", "Phủ định CREATE phải không gọi tool."),
        ("no_action_negated_save", "Không lưu câu gọi điện cho chị Lan nhé.", "Tôi sẽ không lưu câu đó.", "Phủ định CREATE phải không gọi tool."),
        ("no_action_negated_complete", "Đừng đánh dấu task 188 là hoàn thành.", "Tôi sẽ giữ nguyên trạng thái task 188.", "Phủ định COMPLETE phải không gọi tool."),
        ("no_action_negated_done", "Task 230 chưa xong, không được hoàn tất nó.", "Tôi sẽ giữ nguyên trạng thái task 230.", "Phủ định COMPLETE phải không gọi tool."),
        ("no_action_negated_edit", "Không sửa nội dung việc 344 thành mua sữa.", "Tôi sẽ giữ nguyên nội dung việc 344.", "Phủ định EDIT phải không gọi tool."),
        ("no_action_negated_update", "Đừng update task 455 thành gọi Mai.", "Tôi sẽ giữ nguyên nội dung task 455.", "Phủ định EDIT phải không gọi tool."),
        ("no_action_negated_delete", "Không xóa việc 566.", "Tôi sẽ giữ nguyên việc 566.", "Phủ định DELETE phải không gọi tool."),
        ("no_action_negated_remove", "Đừng bỏ task 677 khỏi danh sách.", "Tôi sẽ giữ nguyên task 677.", "Phủ định DELETE phải không gọi tool."),
        ("no_action_negated_deadline", "Không đặt hạn cho việc 788 vào 8 giờ sáng mai.", "Tôi sẽ giữ nguyên deadline của việc 788.", "Phủ định DEADLINE phải không gọi tool."),
        ("no_action_negated_due_change", "Đừng đổi deadline task 809 sang 09:00 ngày mai.", "Tôi sẽ giữ nguyên deadline của task 809.", "Phủ định thay deadline phải không gọi tool."),
        ("no_action_negated_list", "Đừng mở danh sách task của tôi.", "Tôi sẽ không mở danh sách.", "Phủ định LIST phải không gọi tool."),
        ("no_action_negated_due_query", "Không cần xem các việc quá hạn.", "Tôi sẽ không truy vấn các việc quá hạn.", "Phủ định DUE query phải không gọi tool."),
        ("no_action_unsupported_reminder", "Nhắc tôi uống thuốc mỗi tối lúc 9 giờ.", "NEXUS chưa hỗ trợ nhắc việc định kỳ.", "Reminder nằm ngoài phạm vi hiện tại."),
        ("no_action_unsupported_priority", "Sắp các task theo độ ưu tiên rồi gắn nhãn khẩn cấp.", "NEXUS chưa hỗ trợ độ ưu tiên hoặc sắp xếp theo độ ưu tiên.", "Priority và sort nằm ngoài phạm vi hiện tại."),
    ]
    return [_reply(suffix, "no_action", None, prompt, response, rationale) for suffix, prompt, response, rationale in specs]


def build_document() -> dict[str, Any]:
    sanity = json.loads(SANITY_DATASET.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "dataset_id": "nexus-sft-pilot-v1",
        "purpose": "Reviewed 100-example SFT pilot; preserves sanity cases and adds balanced task-agent behavior without using evaluation labels.",
        "base_model": deepcopy(sanity["base_model"]),
        "contract": deepcopy(sanity["contract"]),
        "cases": deepcopy(sanity["cases"]) + _new_call_cases() + _clarification_cases() + _no_action_cases(),
    }


def render_document() -> str:
    return json.dumps(build_document(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true", help="Write the deterministic artifact")
    parser.add_argument("--check", action="store_true", help="Fail if the tracked artifact differs")
    args = parser.parse_args()
    rendered = render_document()
    if args.write:
        args.output.write_text(rendered, encoding="utf-8")
    if args.check and args.output.read_text(encoding="utf-8") != rendered:
        parser.error(f"Generated dataset differs from {args.output}")
    print(json.dumps({"path": str(args.output), "cases": len(build_document()["cases"]), "bytes": len(rendered.encode("utf-8"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
