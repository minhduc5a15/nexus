"""Build the reviewed 400-case NEXUS SFT candidate dataset deterministically."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PILOT_DATASET = ROOT / "evals/sft_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "evals/sft_candidate_v2.json"


def _tool_call(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def _call(
    case_id: str,
    family: str,
    action: str,
    prompt: str,
    tool: str,
    arguments: dict[str, Any],
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "behavior": "call",
        "action": action,
        "rationale": rationale,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": _tool_call(tool, arguments)},
        ],
    }


def _reply(
    case_id: str,
    family: str,
    behavior: str,
    action: str | None,
    prompt: str,
    response: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "behavior": behavior,
        "action": action,
        "rationale": rationale,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def _cross_calls(
    *,
    action: str,
    tool: str,
    frames: Iterable[str],
    values: Iterable[dict[str, Any]],
    arguments,
    rationale: str,
) -> list[dict[str, Any]]:
    cases = []
    for frame_index, frame in enumerate(frames, 1):
        family = f"candidate_{action}_frame_{frame_index:02d}"
        for value_index, value in enumerate(values, 1):
            prompt = frame.format(**value)
            cases.append(
                _call(
                    f"candidate_{action}_{frame_index:02d}_{value_index:02d}",
                    family,
                    action,
                    prompt,
                    tool,
                    arguments(value),
                    rationale,
                )
            )
    return cases


def _create_cases() -> list[dict[str, Any]]:
    frames = [
        "Hãy thêm nguyên văn mục sau vào sổ:\n{content}",
        "Tạo một task với nội dung nằm sau dấu mũi tên → {content}",
        "Ghi giúp mình đúng phần trong ngoặc nhọn: {{{content}}}",
        "Mở một mục mới, giữ nguyên cả chữ và dấu:\n{content}",
        "Please create một task đúng như sau:\n{content}",
        "Đưa đoạn sau vào danh sách mà không diễn đạt lại: {content}",
        "Tạo task từ khối dữ liệu bên dưới.\n---\n{content}",
        "Lưu mục công việc này nguyên xi: «{content}»",
    ]
    values = [
        {"content": "Kiểm lại báo giá, bản cuối."},
        {"content": "gọi cô Hà lúc 16:20"},
        {"content": "soát mục A\n\nsoát mục B"},
        {"content": "đừng quên xóa cache staging"},
    ]
    return _cross_calls(
        action="create",
        tool="create_task",
        frames=frames,
        values=values,
        arguments=lambda value: {"content": value["content"]},
        rationale="CREATE rõ ràng; chép nguyên văn payload, kể cả dấu câu và xuống dòng.",
    )


def _list_cases() -> list[dict[str, Any]]:
    frames = [
        "Mở {ledger} để tôi kiểm tra.",
        "Cho tôi đọc lại {ledger} nhé.",
        "Hiện trong {ledger} đang có những gì?",
        "Show toàn bộ {ledger} cho mình.",
        "Liệt kê nguyên {ledger}, không lọc trạng thái.",
        "Tôi cần rà lại {ledger} từ đầu đến cuối.",
        "Có thể đưa {ledger} hiện tại ra đây không?",
        "Hãy hiển thị tất cả nội dung thuộc {ledger}.",
    ]
    values = [
        {"ledger": "sổ công việc"},
        {"ledger": "danh sách task đã lưu"},
        {"ledger": "các đầu việc của tôi"},
        {"ledger": "toàn bộ mục trong NEXUS"},
    ]
    return _cross_calls(
        action="list",
        tool="list_tasks",
        frames=frames,
        values=values,
        arguments=lambda _value: {},
        rationale="LIST toàn bộ rõ ràng, không có bộ lọc deadline và không có arguments.",
    )


def _complete_cases() -> list[dict[str, Any]]:
    frames = [
        "Công việc mã {id} đã xử lý xong, ghi nhận giúp tôi.",
        "Chuyển trạng thái của task #{id} sang hoàn tất.",
        "Mục số {id} xong rồi nha, đánh dấu lại giùm.",
        "Please mark việc ID {id} as completed.",
        "Tôi xác nhận đầu việc {id} đã hoàn thành.",
        "Cập nhật riêng task mang mã {id} thành đã xong.",
        "Khép lại việc #{id} với trạng thái hoàn thành.",
        "Việc có ID {id} tôi làm xong rồi; ghi trạng thái đó vào sổ.",
    ]
    values = [{"id": value} for value in (17, 204, 583, 941)]
    return _cross_calls(
        action="complete",
        tool="complete_task",
        frames=frames,
        values=values,
        arguments=lambda value: {"id": value["id"]},
        rationale="COMPLETE đúng một ID được người dùng chỉ định.",
    )


def _edit_cases() -> list[dict[str, Any]]:
    frames = [
        "Viết lại nội dung task {id} thành: {content}",
        "Thay phần chữ của mục #{id} bằng đúng câu sau:\n{content}",
        "Update việc ID {id} thành {content}",
        "Mục {id} cần sửa nội dung mới là «{content}»",
        "Giữ nguyên mã {id}, đổi task sang: {content}",
        "Chỉnh riêng đầu việc #{id}; nội dung thay thế:\n{content}",
        "Nội dung cũ của task {id} không còn đúng, hãy ghi lại là {content}",
        "Sửa bản ghi công việc mang ID {id} thành đúng nguyên văn: {content}",
    ]
    values = [
        {"id": 26, "content": "Gửi biên lai cho phòng tài chính."},
        {"id": 318, "content": "mua cáp USB-C, dài 2 m"},
        {"id": 607, "content": "Ôn Rust: ownership & borrow"},
        {"id": 952, "content": "kiểm tra API\n\nviết ghi chú triển khai"},
    ]
    return _cross_calls(
        action="edit",
        tool="update_task",
        frames=frames,
        values=values,
        arguments=lambda value: {"id": value["id"], "content": value["content"]},
        rationale="EDIT giữ đúng một ID và toàn bộ payload mới nguyên văn.",
    )


def _delete_cases() -> list[dict[str, Any]]:
    frames = [
        "Xóa vĩnh viễn mục mang mã {id} khỏi sổ.",
        "Loại task #{id} ra khỏi danh sách giúp tôi.",
        "Tôi không cần đầu việc {id} nữa; hãy xóa nó.",
        "Delete riêng task có ID {id}.",
        "Bỏ bản ghi công việc số {id} khỏi NEXUS.",
        "Cho mục #{id} rời khỏi danh sách hoàn toàn.",
        "Yêu cầu xóa task {id}; ứng dụng cứ hỏi xác nhận sau.",
        "Gỡ vĩnh viễn đầu việc mang ID {id}.",
    ]
    values = [{"id": value} for value in (39, 226, 614, 973)]
    return _cross_calls(
        action="delete",
        tool="delete_task",
        frames=frames,
        values=values,
        arguments=lambda value: {"id": value["id"]},
        rationale="DELETE chỉ đề xuất đúng một ID; xác nhận và side effect thuộc ứng dụng.",
    )


def _deadline_cases() -> list[dict[str, Any]]:
    frames = [
        "Đặt hạn của việc {id} vào {when}.",
        "Task #{id} cần deadline mới là {when}.",
        "Gán mốc {when} cho đầu việc ID {id}.",
        "Với mục {id}, hãy dùng nguyên cụm {when} làm thời hạn.",
        "Set deadline task {id} thành {when}.",
        "Cập nhật hạn riêng cho việc #{id}: {when}.",
        "Mốc chót của bản ghi {id} là {when}; lưu giúp tôi.",
        "Cho task mang ID {id} thời hạn {when}.",
    ]
    values = [
        {"id": 48, "when": "06:40 hôm nay"},
        {"id": 337, "when": "9 giờ 15 phút sáng mai"},
        {"id": 672, "when": "20:05 ngày 03/12/2027"},
        {"id": 986, "when": "14/02/2028 23:10"},
    ]
    return _cross_calls(
        action="deadline",
        tool="set_task_deadline",
        frames=frames,
        values=values,
        arguments=lambda value: {"id": value["id"], "when": value["when"]},
        rationale="DEADLINE giữ đúng ID và cụm thời gian nguyên văn; model không đổi sang timestamp.",
    )


def _due_cases() -> list[dict[str, Any]]:
    prompts = {
        "today": [
            "Lọc các việc có hạn rơi đúng hôm nay.",
            "Trong ngày hôm nay task nào tới hạn?",
            "Cho mình xem phần deadline của hôm nay thôi.",
            "Show những đầu việc đến hạn hôm nay.",
            "Có mục chưa xong nào mang hạn hôm nay không?",
            "Mở danh sách deadline thuộc ngày hôm nay.",
            "Tôi cần rà các task phải tới hạn hôm nay.",
            "Truy vấn phạm vi hôm nay trong lịch hạn.",
            "Những việc nào có mốc chót trong hôm nay?",
            "Liệt kê task đến kỳ xử lý hôm nay.",
            "Hạn trong hôm nay gồm các mục nào?",
        ],
        "tomorrow": [
            "Lọc các việc có hạn vào ngày mai.",
            "Ngày mai task nào sẽ tới hạn?",
            "Cho mình xem riêng deadline của mai.",
            "Show những đầu việc đến hạn ngày mai.",
            "Có mục chưa xong nào mang hạn mai không?",
            "Mở danh sách deadline thuộc ngày mai.",
            "Tôi cần rà các task phải tới hạn vào mai.",
            "Truy vấn phạm vi ngày mai trong lịch hạn.",
            "Những việc nào có mốc chót trong ngày mai?",
            "Liệt kê task đến kỳ xử lý ngày mai.",
            "Hạn của mai gồm các mục nào?",
        ],
        "overdue": [
            "Lọc các việc đã quá hạn.",
            "Task nào đang trễ hạn vậy?",
            "Cho mình xem riêng các deadline đã trễ hạn.",
            "Show những đầu việc quá hạn.",
            "Có mục chưa xong nào đã quá hạn qua mốc chót không?",
            "Mở danh sách task bị trễ hạn.",
            "Tôi cần rà các việc quá hạn vì đã lỡ deadline.",
            "Truy vấn phạm vi overdue, tức quá hạn, trong lịch hạn.",
            "Những việc nào đã quá hạn và đi qua thời hạn?",
            "Liệt kê task trễ hạn vì đã qua mốc xử lý.",
        ],
    }
    cases = []
    index = 0
    for scope, scope_prompts in prompts.items():
        for within_scope, prompt in enumerate(scope_prompts, 1):
            index += 1
            cases.append(
                _call(
                    f"candidate_due_{index:02d}",
                    f"candidate_due_{scope}_frame_{within_scope:02d}",
                    "due_query",
                    prompt,
                    "list_tasks_by_deadline",
                    {"scope": scope},
                    "DUE query chỉ đọc, ánh xạ đúng scope được nêu rõ.",
                )
            )
    return cases


def _clarification_cases() -> list[dict[str, Any]]:
    specs = [
        ("create_unsaid", "create", "Tạo một task mới, lát nữa tôi mới nói nội dung.", "Bạn muốn nội dung task là gì?", "CREATE thiếu content."),
        ("create_placeholder", "create", "Ghi giúp tôi một việc nhưng tôi chưa nghĩ ra câu chữ.", "Bạn muốn ghi nội dung nào vào task?", "CREATE thiếu payload đã chốt."),
        ("create_only_verb", "create", "Lưu một đầu việc giúp mình.", "Bạn muốn lưu đầu việc gì?", "CREATE không có content."),
        ("create_blank_quote", "create", "Thêm task có nội dung để trống.", "Bạn muốn task chứa nội dung gì?", "Task rỗng không hợp lệ."),
        ("complete_no_id", "complete", "Ghi nhận một task là đã làm xong.", "Bạn muốn hoàn thành task có ID nào?", "COMPLETE thiếu ID."),
        ("complete_two", "complete", "Chốt hoàn thành việc 81 và 82.", "Bạn muốn hoàn thành việc 81 hay 82? Hãy chọn đúng một ID.", "COMPLETE có hai ID."),
        ("complete_choice", "complete", "Mark task 144 hoặc 145 complete.", "Bạn muốn hoàn thành task 144 hay 145? Hãy chọn đúng một ID.", "COMPLETE không tự chọn ID."),
        ("complete_many", "complete", "Ba việc 211, 212, 213 đều xong rồi.", "Bạn muốn hoàn thành việc có ID nào trước? Hãy chọn đúng một ID.", "Một lượt chỉ COMPLETE một ID."),
        ("complete_range", "complete", "Hoàn tất các task từ 300 đến 305.", "Bạn muốn hoàn thành task có ID nào? Mỗi lượt chỉ hỗ trợ một ID.", "Không hỗ trợ dải ID."),
        ("edit_no_id", "edit", "Đổi một task thành đặt lịch khám.", "Bạn muốn sửa task có ID nào?", "EDIT thiếu ID."),
        ("edit_no_content", "edit", "Sửa nội dung của việc 421 giúp tôi.", "Bạn muốn đổi nội dung việc 421 thành gì?", "EDIT thiếu content."),
        ("edit_two_ids", "edit", "Viết lại task 431 và 432 thành gọi bên vận chuyển.", "Bạn muốn sửa task 431 hay 432? Hãy chọn đúng một ID.", "EDIT có nhiều ID."),
        ("edit_two_contents", "edit", "Đổi việc 441 thành mua vé hoặc đặt khách sạn.", "Bạn muốn nội dung mới của việc 441 là “mua vé” hay “đặt khách sạn”?", "EDIT có hai payload."),
        ("edit_pending_wording", "edit", "Task 451 cần sửa, nhưng để tôi nghĩ lại nội dung.", "Bạn muốn đổi nội dung task 451 thành gì?", "EDIT chưa có payload."),
        ("edit_batch", "edit", "Cập nhật hàng loạt task 461, 462 và 463.", "Bạn muốn sửa task có ID nào trước, và nội dung mới là gì?", "EDIT batch và thiếu content."),
        ("delete_no_id", "delete", "Xóa một việc không cần thiết khỏi sổ.", "Bạn muốn xóa task có ID nào?", "DELETE thiếu ID."),
        ("delete_two", "delete", "Xóa task 501 với 502.", "Bạn muốn xóa task 501 hay 502? Hãy chọn đúng một ID.", "DELETE có hai ID."),
        ("delete_choice", "delete", "Bỏ mục 511 hoặc 512 đi.", "Bạn muốn xóa mục 511 hay 512? Hãy chọn đúng một ID.", "DELETE không tự chọn ID."),
        ("delete_batch", "delete", "Dọn các task 521, 522, 523 khỏi danh sách.", "Bạn muốn xóa task có ID nào trước? Mỗi lượt chỉ hỗ trợ một ID.", "Không hỗ trợ DELETE batch."),
        ("delete_unknown_reference", "delete", "Xóa cái task hồi nãy tôi vừa nhắc.", "Bạn muốn xóa task có ID nào?", "Không suy ID từ tham chiếu hội thoại."),
        ("deadline_no_id", "deadline", "Đặt hạn 7 giờ sáng mai cho một task.", "Bạn muốn đặt hạn cho task có ID nào?", "DEADLINE thiếu ID."),
        ("deadline_no_time", "deadline", "Gán deadline mới cho việc 601.", "Bạn muốn đặt hạn cho việc 601 vào thời gian nào?", "DEADLINE thiếu when."),
        ("deadline_two_ids", "deadline", "Đặt hạn việc 611 và 612 vào 18:00 ngày mai.", "Bạn muốn đặt hạn cho việc 611 hay 612? Hãy chọn đúng một ID.", "DEADLINE có hai ID."),
        ("deadline_two_times", "deadline", "Task 621 để hạn 8 giờ hoặc 9 giờ sáng mai.", "Bạn muốn đặt hạn task 621 vào 8 giờ hay 9 giờ sáng mai?", "DEADLINE có hai thời điểm."),
        ("deadline_batch", "deadline", "Gán 20:00 hôm nay cho task 631, 632 và 633.", "Bạn muốn đặt hạn cho task có ID nào trước?", "Không hỗ trợ DEADLINE batch."),
        ("deadline_unsaid", "deadline", "Task 641 cần một deadline nhưng tôi chưa chọn giờ.", "Bạn muốn đặt hạn task 641 vào thời gian nào?", "DEADLINE chưa có when."),
        ("deadline_id_choice", "deadline", "Mốc 10:00 mai dành cho task 651 hoặc 652.", "Bạn muốn đặt mốc đó cho task 651 hay 652? Hãy chọn đúng một ID.", "DEADLINE không tự chọn ID."),
        ("deadline_time_choice", "deadline", "Đặt hạn việc 661 vào sáng mai hoặc tối mai.", "Bạn muốn đặt hạn việc 661 vào thời gian cụ thể nào?", "Cụm thời gian có hai lựa chọn mơ hồ."),
        ("due_no_scope", "due_query", "Lọc task theo deadline cho tôi.", "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay đã quá hạn?", "DUE thiếu scope."),
        ("due_unspecified_day", "due_query", "Cho mình xem các việc sắp tới hạn.", "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay đã quá hạn?", "Sắp tới chưa ánh xạ vào scope hỗ trợ."),
        ("due_two_scopes", "due_query", "Xem deadline hôm nay và ngày mai.", "Bạn muốn xem phạm vi hôm nay hay ngày mai? Hãy chọn một phạm vi.", "Một call chỉ có một scope."),
        ("due_choice", "due_query", "Lọc việc quá hạn hoặc đến hạn hôm nay giúp tôi.", "Bạn muốn xem việc quá hạn hay đến hạn hôm nay?", "DUE không tự chọn scope."),
        ("due_generic", "due_query", "Mở phần công việc theo hạn.", "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay đã quá hạn?", "DUE framing rõ nhưng thiếu scope."),
        ("list_or_due", "list", "Cho tôi xem danh sách, có thể toàn bộ hoặc chỉ việc quá hạn.", "Bạn muốn xem toàn bộ danh sách hay chỉ các việc quá hạn?", "Không tự chọn LIST hay DUE."),
        ("list_or_today", "list", "Mở task cho tôi, nhưng chưa biết xem hết hay xem hạn hôm nay.", "Bạn muốn xem toàn bộ task hay chỉ các task đến hạn hôm nay?", "Yêu cầu đọc có hai phạm vi."),
        ("edit_missing_both", "edit", "Tôi cần chỉnh lại một việc.", "Bạn muốn sửa task có ID nào và đổi nội dung thành gì?", "EDIT thiếu cả ID và content."),
        ("deadline_missing_both", "deadline", "Tôi muốn thêm thời hạn cho một việc.", "Bạn muốn đặt hạn cho task có ID nào và vào thời gian nào?", "DEADLINE thiếu ID và when."),
        ("delete_many_words", "delete", "Xóa hai task số bảy trăm lẻ một và bảy trăm lẻ hai.", "Bạn muốn xóa task nào? Hãy cung cấp đúng một ID bằng chữ số.", "Nhiều ID viết bằng chữ không được tự suy diễn."),
    ]
    return [
        _reply(
            f"candidate_clarify_{suffix}",
            f"candidate_clarify_family_{suffix}",
            "clarify",
            action,
            prompt,
            response,
            rationale,
        )
        for suffix, action, prompt, response, rationale in specs
    ]


def _no_action_cases() -> list[dict[str, Any]]:
    specs = [
        ("bare_plan", "Chiều nay ghé lấy hồ sơ.", "Nếu bạn muốn lưu việc này, hãy yêu cầu tôi thêm nó vào danh sách.", "Câu trần thuật không cấp quyền CREATE."),
        ("bare_done", "Tôi vừa gửi báo cáo xong.", "Tôi chưa thay đổi task nào.", "Tường thuật đã làm xong không chỉ ra ID."),
        ("bare_today", "Hôm nay nhiều việc thật.", "Bạn chưa yêu cầu xem hoặc thay đổi task nào.", "Từ hôm nay không tự tạo DUE query."),
        ("bare_question", "Không biết còn kịp mua quà không nhỉ?", "Nếu bạn muốn lưu một task, hãy nói rõ nội dung cần thêm.", "Câu hỏi suy nghĩ không phải action."),
        ("ambiguous_today", "Nay tôi phải để ý chuyện gì vậy?", "Bạn muốn xem toàn bộ task hay các task đến hạn hôm nay?", "Không tự chọn LIST hoặc DUE."),
        ("ambiguous_tomorrow", "Mai có gì cần chú ý?", "Bạn muốn hỏi về task đến hạn ngày mai hay một thông tin khác?", "Không tự suy DUE từ từ mai."),
        ("negated_create", "Không ghi việc gọi ngân hàng vào NEXUS.", "Tôi sẽ không tạo task đó.", "CREATE bị phủ định."),
        ("negated_list", "Chưa cần mở danh sách công việc.", "Tôi sẽ không mở danh sách.", "LIST bị phủ định."),
        ("negated_complete", "Task 710 vẫn chưa xong, đừng hoàn thành nó.", "Tôi sẽ giữ nguyên trạng thái task 710.", "COMPLETE bị phủ định."),
        ("negated_edit", "Không đổi task 720 thành mua mực in.", "Tôi sẽ giữ nguyên nội dung task 720.", "EDIT bị phủ định."),
        ("negated_delete", "Đừng xóa mục 730 dù nó đã cũ.", "Tôi sẽ giữ nguyên mục 730.", "DELETE bị phủ định."),
        ("negated_deadline", "Không đặt mốc 8 giờ mai cho việc 740.", "Tôi sẽ giữ nguyên deadline của việc 740.", "DEADLINE bị phủ định."),
        ("negated_due", "Không cần kiểm tra task quá hạn.", "Tôi sẽ không truy vấn task quá hạn.", "DUE query bị phủ định."),
        ("unsupported_uncomplete", "Bỏ trạng thái hoàn thành của task 750.", "NEXUS chưa hỗ trợ chuyển task đã hoàn thành về chưa hoàn thành.", "Không biến thao tác đảo COMPLETE thành COMPLETE."),
        ("unsupported_reopen", "Reopen việc #751 giúp mình.", "NEXUS chưa hỗ trợ mở lại task đã hoàn thành.", "Reopen nằm ngoài contract."),
        ("unsupported_clear_deadline", "Xóa deadline của task 760.", "NEXUS chưa hỗ trợ gỡ deadline khỏi task.", "Không biến xóa deadline thành SET DEADLINE."),
        ("unsupported_deadline_none", "Set deadline việc 761 về none.", "NEXUS chưa hỗ trợ gỡ deadline khỏi task.", "Giá trị none không phải cụm thời gian hợp lệ."),
        ("unsupported_week", "Đặt hạn task 770 vào tuần sau.", "NEXUS chưa hỗ trợ cụm thời gian “tuần sau”.", "Không tự đổi tuần sau thành tomorrow."),
        ("unsupported_weekday", "Đặt hạn việc 771 vào thứ Hai tới.", "NEXUS chưa hỗ trợ deadline theo thứ trong tuần.", "Grammar chưa hỗ trợ weekday."),
        ("unsupported_date_only", "Đặt hạn task 772 vào ngày 20/12/2027.", "NEXUS cần cả ngày và giờ để đặt deadline.", "Ngày thiếu giờ không được gọi tool."),
        ("unsupported_time_only", "Đặt hạn việc 773 lúc 15:00.", "NEXUS cần cả ngày và giờ để đặt deadline.", "Giờ thiếu ngày không được gọi tool."),
        ("unsupported_recurrence", "Mỗi thứ Sáu tạo lại task kiểm tra backup.", "NEXUS chưa hỗ trợ task lặp lại.", "Recurrence nằm ngoài phạm vi."),
        ("unsupported_reminder", "Nhắc tôi trước deadline task 780 khoảng 30 phút.", "NEXUS chưa hỗ trợ nhắc việc.", "Reminder không phải DEADLINE."),
        ("unsupported_priority", "Đánh dấu việc 781 là ưu tiên cao.", "NEXUS chưa hỗ trợ độ ưu tiên.", "Priority nằm ngoài phạm vi."),
        ("unsupported_tags", "Gắn nhãn công việc cho task 782.", "NEXUS chưa hỗ trợ nhãn task.", "Tag nằm ngoài phạm vi."),
        ("unsupported_search_content", "Tìm task có chữ hóa đơn rồi xóa nó.", "NEXUS chỉ xóa theo một ID cụ thể và chưa hỗ trợ tìm để tự chọn task.", "Không suy ID từ content."),
        ("unsupported_complete_content", "Hoàn thành task mang nội dung gọi mẹ.", "NEXUS chỉ hoàn thành task theo một ID cụ thể.", "Không tìm COMPLETE theo content."),
        ("unsupported_edit_content_lookup", "Sửa việc mua sữa thành mua bánh.", "NEXUS chỉ sửa task khi bạn cung cấp một ID cụ thể.", "Không tìm EDIT theo content."),
        ("unsupported_delete_last", "Xóa việc vừa tạo đi.", "NEXUS chưa hỗ trợ tham chiếu “việc vừa tạo”; hãy dùng ID cụ thể.", "Không dùng tham chiếu hội thoại để DELETE."),
        ("unsupported_due_week", "Cho tôi xem các task đến hạn tuần sau.", "NEXUS chỉ hỗ trợ truy vấn hạn hôm nay, ngày mai hoặc quá hạn.", "DUE chưa hỗ trợ tuần sau."),
        ("quoted_create", "Ví dụ người dùng có thể nói: “Thêm việc mua sữa”.", "Đó là một câu ví dụ; tôi chưa tạo task nào.", "Lời trích dẫn không cấp quyền CREATE."),
        ("quoted_delete", "Câu ‘xóa task 800’ nghe có rõ không?", "Câu đó rõ về mặt diễn đạt, nhưng tôi chưa xóa task nào.", "Thảo luận về câu lệnh không phải yêu cầu DELETE."),
        ("capability_question", "Bạn có thể đặt deadline theo ID không?", "Có. NEXUS có thể đặt deadline cho một task khi bạn cung cấp đúng một ID và cụm thời gian được hỗ trợ.", "Hỏi khả năng không thực thi tool."),
        ("explain_list", "LIST trong NEXUS nghĩa là gì?", "LIST là thao tác xem toàn bộ task đã lưu.", "Hỏi khái niệm không gọi LIST."),
        ("invalid_deadline_date", "Đặt hạn task 810 vào 31/02/2028 10:00.", "Ngày 31/02/2028 không hợp lệ nên tôi chưa đặt deadline.", "Ngày không tồn tại không được gọi tool."),
        ("invalid_deadline_hour", "Đặt hạn task 811 vào 29:30 ngày mai.", "Giờ 29:30 không hợp lệ nên tôi chưa đặt deadline.", "Giờ ngoài phạm vi không được gọi tool."),
        ("unsupported_multi_create_semantics", "Từ câu ‘ăn sáng và tập thể dục’, tự tách thành hai task giúp tôi.", "NEXUS chưa tự tách nhiều task từ một dòng; mỗi dòng mới tạo một task.", "Không tự phân tách theo từ và."),
        ("unsupported_undo_delete", "Khôi phục task 812 vừa bị xóa.", "NEXUS chưa hỗ trợ hoàn tác việc xóa task.", "Undo DELETE nằm ngoài phạm vi."),
    ]
    return [
        _reply(
            f"candidate_no_action_{suffix}",
            f"candidate_no_action_family_{suffix}",
            "no_action",
            None,
            prompt,
            response,
            rationale,
        )
        for suffix, prompt, response, rationale in specs
    ]


def build_document() -> dict[str, Any]:
    pilot = json.loads(PILOT_DATASET.read_text(encoding="utf-8"))
    new_cases = (
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
        "dataset_id": "nexus-sft-candidate-v2",
        "purpose": "Reviewed 400-example SFT candidate dataset; expands the frozen pilot with decision-boundary coverage without using evaluation labels.",
        "base_model": deepcopy(pilot["base_model"]),
        "contract": deepcopy(pilot["contract"]),
        "cases": deepcopy(pilot["cases"]) + new_cases,
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
