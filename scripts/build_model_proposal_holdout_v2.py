"""Build the blind 120-case NEXUS final proposal holdout v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evals/model_proposal_holdout_v2.json"


def _call(
    case_id: str,
    family: str,
    action: str,
    prompt: str,
    tool: str,
    arguments: dict[str, Any],
    rationale: str,
    dimensions: list[str],
) -> dict[str, Any]:
    accepted_calls = [{"name": tool, "arguments": arguments}]
    content = arguments.get("content")
    if (
        action in {"create", "edit"}
        and isinstance(content, str)
        and content
        and content[-1] not in ".?!"
        and f"{content}." in prompt
    ):
        punctuated = dict(arguments)
        punctuated["content"] = f"{content}."
        accepted_calls.append({"name": tool, "arguments": punctuated})
    return {
        "id": case_id,
        "prompt": prompt,
        "family": family,
        "rationale": rationale,
        "dimensions": dimensions,
        "origin": "new_authored_v2",
        "expected": {
            "behavior": "call",
            "action": action,
            "accepted_calls": accepted_calls,
            "missing_slots": [],
        },
    }


def _control(
    case_id: str,
    family: str,
    behavior: str,
    action: str | None,
    prompt: str,
    rationale: str,
    missing_slots: list[str],
    dimensions: list[str],
) -> dict[str, Any]:
    case = {
        "id": case_id,
        "prompt": prompt,
        "family": family,
        "rationale": rationale,
        "dimensions": dimensions,
        "origin": "new_authored_v2",
        "expected": {
            "behavior": behavior,
            "action": action,
            "accepted_calls": [],
            "missing_slots": missing_slots,
        },
    }
    if behavior == "clarify":
        case["known_fields"] = {}
    return case


def _cross_calls(
    *,
    prefix: str,
    action: str,
    tool: str,
    frames: Iterable[str],
    values: Iterable[dict[str, Any]],
    arguments: Callable[[dict[str, Any]], dict[str, Any]],
    rationale: str,
    dimensions: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for frame_index, frame in enumerate(frames, 1):
        for value_index, value in enumerate(values, 1):
            index = (frame_index - 1) * 4 + value_index
            rows.append(
                _call(
                    f"final2_{prefix}_{index:02d}",
                    f"final2_{prefix}_family_{frame_index:02d}",
                    action,
                    frame.format(**value),
                    tool,
                    arguments(value),
                    rationale,
                    dimensions,
                )
            )
    return rows


def _create_cases() -> list[dict[str, Any]]:
    specs = [
        ("Thêm giúp tôi việc đo lại chiều rộng cái kệ.", "đo lại chiều rộng cái kệ"),
        ("Ghi lại việc nhắn anh Khoa hỏi mã QR bị lỗi nhé.", "nhắn anh Khoa hỏi mã QR bị lỗi"),
        ("Cho tôi thêm task kiểm tra bản sao lưu tối nay.", "kiểm tra bản sao lưu tối nay"),
        ("Note giúp mình: gọi thợ sửa máy lạnh", "gọi thợ sửa máy lạnh"),
        ("Thêm hai việc này vào danh sách, mỗi dòng một việc:\nđọc chương 3\nlàm bài tập", "đọc chương 3\nlàm bài tập"),
        ("Tạo việc mua cáp USB-C, loại dài 2 mét.", "mua cáp USB-C, loại dài 2 mét"),
        ("Lưu giúp tôi việc ôn C++: std::span.", "ôn C++: std::span"),
        ("Nhớ ghi vào sổ cho tôi: đừng quên xóa cache staging", "đừng quên xóa cache staging"),
        ("Quăng giúp tui việc mang pin cũ đi tái chế vào danh sách.", "mang pin cũ đi tái chế"),
        ("Please add task gọi phòng khám vào sáng mai.", "gọi phòng khám vào sáng mai"),
        ("Ghi việc kiểm tra hóa đơn & gửi biên lai giúp tôi.", "kiểm tra hóa đơn & gửi biên lai"),
        ("Thêm task kiểm tra API và viết README, coi đó là một việc thôi.", "kiểm tra API và viết README"),
    ]
    return [
        _call(
            f"final2_create_{index:02d}",
            f"final2_create_natural_family_{index:02d}",
            "create",
            prompt,
            "create_task",
            {"content": content},
            "CREATE rõ ràng bằng cách nói tự nhiên; dấu kết câu của lời yêu cầu không thuộc content.",
            ["blind_family", "natural_language", "content_boundary"],
        )
        for index, (prompt, content) in enumerate(specs, 1)
    ]


def _list_cases() -> list[dict[str, Any]]:
    return _cross_calls(
        prefix="list",
        action="list",
        tool="list_tasks",
        frames=[
            "Xuất nguyên bảng {ledger} hiện tại cho tôi đối chiếu.",
            "Tôi cần nhìn toàn bộ {ledger}, gồm cả mục đã hoàn tất.",
            "Read back mọi bản ghi thuộc {ledger} mà NEXUS đang giữ.",
        ],
        values=[
            {"ledger": "công việc cá nhân"},
            {"ledger": "task của tài khoản này"},
            {"ledger": "sổ các việc đã nhập"},
            {"ledger": "danh mục cần làm"},
        ],
        arguments=lambda _value: {},
        rationale="LIST toàn bộ được nêu rõ, không yêu cầu lọc deadline.",
        dimensions=["blind_family", "read_only"],
    )


def _complete_cases() -> list[dict[str, Any]]:
    return _cross_calls(
        prefix="complete",
        action="complete",
        tool="complete_task",
        frames=[
            "Gạch dấu đã làm xong cho bản ghi mang mã {id}.",
            "Task identity {id} vừa hoàn tất; cập nhật cờ completed cho nó.",
            "Chỉ mục #{id} đã được tôi xử lý xong, ghi nhận trạng thái ấy.",
        ],
        values=[{"id": value} for value in (3107, 4629, 7054, 9368)],
        arguments=lambda value: {"id": value["id"]},
        rationale="COMPLETE đúng một ID được chỉ định.",
        dimensions=["blind_family", "exact_id"],
    )


def _edit_cases() -> list[dict[str, Any]]:
    specs = [
        (3218, "Nội dung mới của việc 3218 là chốt số liệu quý IV.", "chốt số liệu quý IV"),
        (5740, "Task #5740 giờ đổi sang mua đầu chuyển HDMI loại mini.", "mua đầu chuyển HDMI loại mini"),
        (8163, "Việc 8163 sửa lại thành kiểm tra C#: Span<T> nhé.", "kiểm tra C#: Span<T>"),
        (9472, "Cập nhật task 9472 thành đọc hợp đồng và ghi lại điều 7.", "đọc hợp đồng và ghi lại điều 7"),
        (3251, "Sửa đầu việc số 3251 thành gọi lại cho chị Mai.", "gọi lại cho chị Mai"),
        (5864, "Task 5864 đổi thành đặt vé tàu, ghế gần cửa sổ.", "đặt vé tàu, ghế gần cửa sổ"),
        (8290, "Cho việc #8290 nội dung mới là ôn Rust: lifetime.", "ôn Rust: lifetime"),
        (9517, "Từ giờ việc 9517 ghi là gửi ảnh gốc cho khách.", "gửi ảnh gốc cho khách"),
        (3362, "Việc 3362 cần đổi nội dung sang mua bút, vở & thước.", "mua bút, vở & thước"),
        (5987, "Đổi task #5987 thành kiểm tra cổng 8080 lúc chiều.", "kiểm tra cổng 8080 lúc chiều"),
        (8416, "Ghi đè nội dung việc 8416 thành soát tên Nguyễn Ánh.", "soát tên Nguyễn Ánh"),
        (9735, "Sửa task 9735 thành hai dòng này:\nkiểm tra API\nviết ghi chú triển khai", "kiểm tra API\nviết ghi chú triển khai"),
    ]
    return [
        _call(
            f"final2_edit_{index:02d}",
            f"final2_edit_natural_family_{index:02d}",
            "edit",
            prompt,
            "update_task",
            {"id": task_id, "content": content},
            "EDIT nêu đúng một ID và nội dung mới bằng lời nói tự nhiên.",
            ["blind_family", "natural_language", "exact_id", "content_boundary"],
        )
        for index, (task_id, prompt, content) in enumerate(specs, 1)
    ]


def _delete_cases() -> list[dict[str, Any]]:
    prompts = [
        (3329, "Xóa việc 3329 giúp tôi."),
        (4815, "Tôi muốn bỏ hẳn task #4815."),
        (7296, "Tôi không cần việc số 7296 nữa, xóa nó đi."),
        (9584, "Xoá task có ID 9584."),
        (3471, "Cho tôi xóa đầu việc 3471."),
        (4936, "Gỡ việc #4936 ra khỏi danh sách giúp mình."),
        (7418, "Việc 7418 không còn dùng nữa, hãy xóa nó."),
        (9620, "Delete task 9620 giúp tôi."),
        (3584, "Xóa riêng mục số 3584 thôi."),
        (5072, "Bỏ đầu việc mang ID 5072 khỏi sổ."),
        (7539, "Tôi muốn xóa task #7539."),
        (9846, "Xoá việc 9846 khỏi NEXUS nhé."),
    ]
    return [
        _call(
            f"final2_delete_{index:02d}",
            f"final2_delete_natural_family_{index:02d}",
            "delete",
            prompt,
            "delete_task",
            {"id": task_id},
            "DELETE nêu đúng một ID; ứng dụng vẫn hỏi xác nhận sau proposal.",
            ["blind_family", "natural_language", "exact_id", "confirmation_boundary"],
        )
        for index, (task_id, prompt) in enumerate(prompts, 1)
    ]


def _deadline_cases() -> list[dict[str, Any]]:
    specs = [
        (3430, "04:55 hôm nay", "Việc 3430 cần hoàn tất trước 04:55 hôm nay."),
        (5962, "6 giờ 40 phút tối mai", "Cho task 5962 hạn vào 6 giờ 40 phút tối mai."),
        (8374, "17:35 ngày 23/01/2029", "Tôi muốn deadline của task #8374 rơi vào 17:35 ngày 23/01/2029."),
        (9691, "30/06/2029 21:05", "Hạn của việc 9691 là 30/06/2029 21:05."),
        (3542, "07:10 ngày mai", "Gán hạn 07:10 ngày mai cho việc 3542."),
        (6083, "8 giờ 15 phút sáng mai", "Task #6083 cần xong lúc 8 giờ 15 phút sáng mai."),
        (8495, "20:30 ngày 14/03/2029", "Đổi deadline việc 8495 thành 20:30 ngày 14/03/2029."),
        (9713, "01/07/2029 06:25", "Set hạn task 9713 vào 01/07/2029 06:25."),
        (3657, "19:45 hôm nay", "Việc 3657 đặt hạn lúc 19:45 hôm nay nhé."),
        (6194, "9 giờ sáng mai", "Đặt task 6194 tới hạn lúc 9 giờ sáng mai."),
        (8527, "13:20 ngày 02/09/2029", "Cho việc #8527 deadline 13:20 ngày 02/09/2029."),
        (9861, "31/12/2029 23:00", "Cập nhật hạn của task 9861 thành 31/12/2029 23:00."),
    ]
    return [
        _call(
            f"final2_deadline_{index:02d}",
            f"final2_deadline_natural_family_{index:02d}",
            "deadline",
            prompt,
            "set_task_deadline",
            {"id": task_id, "when": when},
            "DEADLINE nêu đúng một ID và một cụm thời gian được hỗ trợ.",
            ["blind_family", "natural_language", "exact_id", "verbatim_time"],
        )
        for index, (task_id, when, prompt) in enumerate(specs, 1)
    ]


def _due_cases() -> list[dict[str, Any]]:
    specs = [
        ("today", "Hôm nay có việc nào đến hạn?"),
        ("today", "Cho tôi xem các task phải xong hôm nay."),
        ("today", "Liệt kê những việc tới hạn trong ngày hôm nay giúp mình."),
        ("today", "Xem deadline hôm nay có những gì."),
        ("tomorrow", "Ngày mai có task nào đến hạn không?"),
        ("tomorrow", "Cho tôi xem những việc phải làm xong vào ngày mai."),
        ("tomorrow", "Liệt kê các đầu việc tới hạn ngày mai."),
        ("tomorrow", "Mai tôi có những deadline nào?"),
        ("overdue", "Có việc nào đang quá hạn không?"),
        ("overdue", "Cho tôi xem danh sách task bị trễ hạn."),
        ("overdue", "Những việc nào đã qua deadline mà chưa xong?"),
        ("overdue", "Liệt kê các đầu việc quá hạn giúp tôi."),
    ]
    return [
        _call(
            f"final2_due_{index:02d}",
            f"final2_due_family_{index:02d}",
            "due_query",
            prompt,
            "list_tasks_by_deadline",
            {"scope": scope},
            "DUE query nêu một scope được hỗ trợ.",
            ["blind_family", "read_only", "exact_scope"],
        )
        for index, (scope, prompt) in enumerate(specs, 1)
    ]


def _clarification_cases() -> list[dict[str, Any]]:
    specs = [
        ("create_payload", "create", "Thêm cho tôi một việc nữa, nội dung để tôi nói sau.", "content", "CREATE chưa có nội dung."),
        ("complete_choice", "complete", "Đánh dấu xong việc 4011 hoặc 4012.", "một id", "Không tự chọn giữa hai ID."),
        ("edit_payload", "edit", "Sửa task 4013 giúp tôi, nội dung mới tôi chưa nghĩ ra.", "content", "EDIT thiếu nội dung mới."),
        ("edit_identity", "edit", "Đổi một task thành kiểm hàng tồn, nhưng tôi quên mất ID.", "id", "EDIT thiếu ID."),
        ("delete_identity", "delete", "Xóa một việc giúp tôi, tôi chưa nhớ số ID.", "id", "DELETE thiếu ID."),
        ("delete_choice", "delete", "Xóa việc 4014 hoặc 4015 cũng được.", "một id", "DELETE không tự chọn ID."),
        ("deadline_identity", "deadline", "Đặt hạn 05:20 ngày mai cho một việc, ID tôi sẽ gửi sau.", "id", "DEADLINE thiếu ID."),
        ("deadline_time", "deadline", "Việc 4016 cần đặt hạn nhưng tôi chưa chọn giờ.", "when", "DEADLINE thiếu thời gian."),
        ("deadline_choice", "deadline", "Task 4017 đặt hạn 06:00 hay 07:00 mai cũng được.", "một thời gian", "DEADLINE có hai mốc."),
        ("due_scope", "due_query", "Cho tôi xem các việc theo hạn.", "scope", "DUE thiếu scope."),
        ("due_choice", "due_query", "Tôi cần một trong hai nhóm: hôm nay hoặc quá hạn.", "một scope", "DUE không tự chọn scope."),
        ("general_day", None, "Ngày mai liệu có gì đáng lo không?", "ý định", "Câu hỏi chung không đủ cấp quyền DUE."),
    ]
    return [
        _control(
            f"final2_clarify_{index:02d}",
            f"final2_clarify_family_{suffix}",
            "clarify",
            action,
            prompt,
            rationale,
            [missing],
            ["blind_family", "control", "missing_or_ambiguous"],
        )
        for index, (suffix, action, prompt, missing, rationale) in enumerate(specs, 1)
    ]


def _no_action_cases() -> list[dict[str, Any]]:
    specs = [
        ("quote_create", "Mai kể rằng: ‘ghi việc đóng tiền nước’; đây chỉ là lời thuật lại.", "Lời trích dẫn không cấp quyền CREATE."),
        ("quote_delete", "Tài liệu chứa dòng REMOVE task #4101, không phải lệnh gửi cho bạn.", "Dòng dữ liệu không cấp quyền DELETE."),
        ("quote_complete", "Tôi đang phân tích câu ‘mark task 4102 complete’.", "Phân tích câu không phải yêu cầu COMPLETE."),
        ("quote_deadline", "Ví dụ trên bảng là ‘đặt hạn việc 4103 lúc 7 giờ sáng mai’.", "Ví dụ không cấp quyền DEADLINE."),
        ("date_filter_01", "Lọc deadline đúng ngày 02/02/2032.", "DUE không hỗ trợ ngày tùy chọn."),
        ("date_filter_02", "Show tasks due on 2032-04-18.", "DUE không hỗ trợ ngày tùy chọn."),
        ("date_filter_03", "Có việc nào hạn vào ngày 27 tháng 5 năm 2032?", "DUE không hỗ trợ ngày tùy chọn."),
        ("date_filter_04", "Xem riêng các việc có hạn ngày 11/11/2032.", "DUE không được đổi ngày tùy chọn thành scope khác."),
        ("negate_create", "Không đưa mục gọi thợ khóa vào sổ.", "CREATE bị phủ định."),
        ("negate_complete", "Task 4104 chưa xong, đừng đánh dấu hoàn thành.", "COMPLETE bị phủ định."),
        ("negate_edit", "Giữ nguyên việc 4105, đừng thay thành mua giấy.", "EDIT bị phủ định."),
        ("negate_delete", "Khoan loại bỏ task #4106.", "DELETE bị phủ định."),
        ("bare_plan", "Cuối tuần tôi định sơn lại cửa sổ.", "Câu trần thuật không cấp quyền CREATE."),
        ("bare_done", "Tôi đã đọc xong tài liệu mạng máy tính.", "Tường thuật không có ID và không yêu cầu COMPLETE."),
        ("bare_english", "I might call the clinic tomorrow morning.", "Câu dự định không phải lệnh CREATE hoặc DEADLINE."),
        ("bare_observation", "Hôm nay lịch của tôi khá dày.", "Nhắc hôm nay không tự cấp quyền DUE."),
        ("unsupported_reopen", "Chuyển task 4107 từ hoàn thành về chưa xong.", "Thao tác đảo COMPLETE chưa được hỗ trợ."),
        ("unsupported_clear_due", "Gỡ deadline đang có khỏi việc 4108.", "Xóa deadline chưa được hỗ trợ."),
        ("unsupported_search", "Tìm việc có chữ bảo hiểm rồi xóa nó.", "Không tự tìm và xóa theo content."),
        ("unsupported_week", "Đưa ra những task tới hạn trong tuần kế tiếp.", "DUE chưa hỗ trợ phạm vi tuần."),
        ("meta_create", "Khi thêm việc thì tôi cần nói những gì?", "Hỏi cách dùng không thực thi CREATE."),
        ("meta_delete", "Tại sao xóa task lại cần xác nhận?", "Hỏi giải thích không thực thi DELETE."),
        ("meta_quote", "Câu ‘sửa task 5 thành gọi Hà’ có nghĩa là gì?", "Giải thích câu trích dẫn không thực thi EDIT."),
        ("meta_capability", "NEXUS có hỗ trợ reminder tự bật không?", "Hỏi khả năng không phải yêu cầu side effect."),
    ]
    return [
        _control(
            f"final2_no_action_{index:02d}",
            f"final2_no_action_family_{suffix}",
            "no_action",
            None,
            prompt,
            rationale,
            [],
            ["blind_family", "control", suffix.split("_")[0]],
        )
        for index, (suffix, prompt, rationale) in enumerate(specs, 1)
    ]


def build_document() -> dict[str, Any]:
    cases = (
        _create_cases()
        + _list_cases()
        + _complete_cases()
        + _edit_cases()
        + _delete_cases()
        + _deadline_cases()
        + _due_cases()
        + _clarification_cases()
        + _no_action_cases()
    )
    return {
        "schema_version": 1,
        "split": "final_holdout",
        "version": "v2",
        "purpose": "Blind final proposal evaluation after SFT dataset v3; never used for training, prompt tuning or checkpoint selection.",
        "label_source": "independently authored gold labels; classifier and policy are not used as oracle",
        "cases": cases,
    }


def render_document() -> str:
    return json.dumps(build_document(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_document()
    if args.write:
        args.output.write_text(rendered, encoding="utf-8")
    if args.check and args.output.read_text(encoding="utf-8") != rendered:
        parser.error(f"Generated holdout differs from {args.output}")
    print(json.dumps({"path": str(args.output), "cases": len(build_document()["cases"]), "bytes": len(rendered.encode())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
