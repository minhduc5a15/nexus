"""Build the 600-case NEXUS SFT v3 dataset deterministically."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from scripts.build_sft_candidate_v2 import build_document as build_v2_document


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evals/sft_candidate_v3.json"


def _call(
    case_id: str,
    family: str,
    action: str,
    prompt: str,
    tool: str,
    arguments: dict[str, Any],
    target: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "behavior": "call",
        "action": action,
        "target_primary": target,
        "rationale": rationale,
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": tool, "arguments": arguments}}
                ],
            },
        ],
    }


def _reply(
    case_id: str,
    family: str,
    behavior: str,
    action: str | None,
    prompt: str,
    response: str,
    target: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "behavior": behavior,
        "action": action,
        "target_primary": target,
        "rationale": rationale,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def _cross(
    *,
    action: str,
    tool: str,
    prefix: str,
    frames: Iterable[str],
    values: Iterable[dict[str, Any]],
    arguments: Callable[[dict[str, Any]], dict[str, Any]],
    target: str,
    rationale: str,
) -> list[dict[str, Any]]:
    cases = []
    for frame_index, frame in enumerate(frames, 1):
        for value_index, value in enumerate(values, 1):
            cases.append(
                _call(
                    f"v3_{prefix}_{frame_index:02d}_{value_index:02d}",
                    f"v3_{prefix}_family_{frame_index:02d}",
                    action,
                    frame.format(**value),
                    tool,
                    arguments(value),
                    target,
                    rationale,
                )
            )
    return cases


def _create_cases() -> list[dict[str, Any]]:
    colloquial = _cross(
        action="create",
        tool="create_task",
        prefix="create_colloquial",
        frames=[
            "Quăng hộ mình vào sổ việc này nha: {content}",
            "Cho mục này vô danh sách cần làm giùm tui — {content}",
            "Nhờ lưu lại thành một đầu việc, nội dung là {content}",
            "Cất câu sau thành task giúp mình: {content}",
        ],
        values=[
            {"content": "đem pin cũ đi tái chế"},
            {"content": "soạn thư hỏi tình trạng đơn #A17"},
            {"content": "đổi mật khẩu Wi-Fi tầng 2"},
            {"content": "kiểm tra bản dịch tiếng Việt"},
        ],
        arguments=lambda value: {"content": value["content"]},
        target="colloquial_create_coverage",
        rationale="Yêu cầu CREATE đời thường nhưng rõ quyền; content phải được chép nguyên văn.",
    )
    exact = _cross(
        action="create",
        tool="create_task",
        prefix="create_character",
        frames=["Đưa nguyên khối giữa BEGIN và END vào một task.\nBEGIN\n{content}\nEND"],
        values=[
            {"content": "Kiểm tra mã XyZ-09."},
            {"content": "soát nhãn: café & trà"},
            {"content": "dòng Một\ndòng hai"},
            {"content": "gọi API /v2/items?id=7"},
        ],
        arguments=lambda value: {"content": value["content"]},
        target="verbatim_character_fidelity",
        rationale="Payload có ký tự dễ bị chuẩn hóa; giữ đúng từng ký tự và xuống dòng.",
    )
    return colloquial + exact


def _list_cases() -> list[dict[str, Any]]:
    return _cross(
        action="list",
        tool="list_tasks",
        prefix="list_balanced",
        frames=[
            "Mở trọn bộ {ledger}; đừng lọc mục đã xong.",
            "Đưa toàn cảnh {ledger} hiện có ra đây giúp tôi.",
            "Tôi muốn rà từ đầu tới cuối {ledger}.",
            "Display mọi mục đang nằm trong {ledger}.",
            "Cho mình một lượt xem đầy đủ {ledger} nhé.",
        ],
        values=[
            {"ledger": "sổ NEXUS"},
            {"ledger": "danh mục công việc"},
            {"ledger": "các task đã ghi"},
            {"ledger": "bảng việc cá nhân"},
        ],
        arguments=lambda _value: {},
        target="balanced_regression_coverage",
        rationale="LIST toàn bộ rõ ràng, không mang scope deadline.",
    )


def _complete_cases() -> list[dict[str, Any]]:
    return _cross(
        action="complete",
        tool="complete_task",
        prefix="complete_balanced",
        frames=[
            "Việc mang số hiệu {id} tôi xử lý xong; cập nhật trạng thái đi.",
            "Tick completed cho đúng mục #{id} giúp mình.",
            "Ghi vào sổ rằng đầu việc ID {id} đã hoàn tất.",
            "Khóa task {id} ở trạng thái làm xong nhé.",
            "Mark riêng bản ghi công việc {id} là complete.",
        ],
        values=[{"id": value} for value in (1203, 2478, 5316, 8842)],
        arguments=lambda value: {"id": value["id"]},
        target="balanced_regression_coverage",
        rationale="COMPLETE đúng một ID do người dùng nêu.",
    )


def _edit_cases() -> list[dict[str, Any]]:
    boundary = _cross(
        action="edit",
        tool="update_task",
        prefix="edit_boundary",
        frames=[
            "Ở bản ghi {id}, thay toàn bộ phần chữ bằng: {content}",
            "Nội dung mới dành cho task #{id} nằm sau ký hiệu DATA> {content}",
            "Giữ ID {id} và chép phần giữa ngoặc vuông làm nội dung mới [{content}]",
            "Rewrite mục số {id}; payload thay thế ở dòng kế:\n{content}",
        ],
        values=[
            {"id": 1357, "content": "Gửi mẫu thử, bản màu xanh."},
            {"id": 2864, "content": "hỏi giá RAM 32 GB!"},
            {"id": 5702, "content": "Đọc mục 4.2?"},
            {"id": 9135, "content": "mua táo, lê & nho."},
        ],
        arguments=lambda value: {"id": value["id"], "content": value["content"]},
        target="verbatim_content_boundary",
        rationale="Dấu câu nằm trong vùng payload được đánh dấu và phải được giữ nguyên.",
    )
    exact = _cross(
        action="edit",
        tool="update_task",
        prefix="edit_character",
        frames=["Sửa task {id} bằng đúng chuỗi sau, kể cả hoa thường: <DATA>{content}</DATA>"],
        values=[
            {"id": 1468, "content": "đối chiếu SKU-aB09"},
            {"id": 3971, "content": "Test cổng :8080"},
            {"id": 6240, "content": "soát tên Nguyễn Ánh"},
            {"id": 9753, "content": "ghi chú C++: std::span"},
        ],
        arguments=lambda value: {"id": value["id"], "content": value["content"]},
        target="verbatim_character_fidelity",
        rationale="EDIT không chuẩn hóa chữ hoa, dấu hoặc ký tự kỹ thuật trong payload.",
    )
    return boundary + exact


def _delete_cases() -> list[dict[str, Any]]:
    return _cross(
        action="delete",
        tool="delete_task",
        prefix="delete_balanced",
        frames=[
            "Gỡ hẳn bản ghi mang số {id}; ứng dụng hỏi lại tôi sau.",
            "Remove đúng task #{id} khỏi sổ công việc.",
            "Tôi yêu cầu loại vĩnh viễn mục ID {id}.",
            "Cho đầu việc {id} biến mất khỏi danh sách, nhớ xác nhận trước.",
            "Xóa riêng record công việc số {id} giúp tôi.",
        ],
        values=[{"id": value} for value in (1582, 3047, 6495, 9271)],
        arguments=lambda value: {"id": value["id"]},
        target="balanced_regression_coverage",
        rationale="DELETE đề xuất đúng một ID; ứng dụng vẫn chịu trách nhiệm xác nhận.",
    )


def _deadline_cases() -> list[dict[str, Any]]:
    exact = _cross(
        action="deadline",
        tool="set_task_deadline",
        prefix="deadline_exact",
        frames=[
            "Đối với task mã {id}, gán nguyên cụm thời gian sau làm hạn: {when}",
            "Mốc chót {when} thuộc về đúng việc #{id}; lưu cả hai giá trị.",
            "Set due time của record {id} bằng chuỗi `{when}`.",
            "Việc ID {id} cần được đặt hạn tại {when}, không diễn giải lại.",
        ],
        values=[
            {"id": 1694, "when": "05:45 ngày mai"},
            {"id": 4128, "when": "7 giờ 05 phút sáng mai"},
            {"id": 7381, "when": "22:10 ngày 09/11/2028"},
            {"id": 9650, "when": "31/12/2028 18:35"},
        ],
        arguments=lambda value: {"id": value["id"], "when": value["when"]},
        target="exact_deadline_id_and_time",
        rationale="ID và when là hai span độc lập, đều phải được chép chính xác.",
    )
    boundary = _cross(
        action="deadline",
        tool="set_task_deadline",
        prefix="deadline_boundary",
        frames=["Đặt hạn cho việc {id}; TIME[{when}] là toàn bộ dữ liệu thời gian."],
        values=[
            {"id": 1786, "when": "06:15 hôm nay"},
            {"id": 4259, "when": "8 giờ tối mai"},
            {"id": 7460, "when": "13:25 ngày 08/10/2028"},
            {"id": 9814, "when": "29/02/2028 09:40"},
        ],
        arguments=lambda value: {"id": value["id"], "when": value["when"]},
        target="verbatim_content_boundary",
        rationale="Chỉ nội dung trong TIME[...] là when; không kéo lời dẫn vào argument.",
    )
    return exact + boundary


def _due_cases() -> list[dict[str, Any]]:
    rows = [
        ("today", "Rà lịch hạn và đưa ra các mục đáo hạn trong hôm nay."),
        ("today", "Tách riêng phần công việc chạm deadline ngày hôm nay."),
        ("today", "Scope today, tức hôm nay, trong due list của NEXUS gồm những task nào?"),
        ("today", "Mở nhóm đầu việc có mốc chót thuộc hôm nay, tức ngày hiện tại."),
        ("today", "Chỉ hiển thị task chưa xong mà hạn rơi vào hôm nay."),
        ("today", "Trong lịch deadline, truy vấn đúng phạm vi hôm nay."),
        ("today", "Tôi cần bảng các việc tới kỳ trong hôm nay."),
        ("tomorrow", "Rà lịch hạn và đưa ra các mục đáo hạn vào ngày mai, tức ngày kế tiếp."),
        ("tomorrow", "Tách riêng phần công việc chạm deadline của ngày mai."),
        ("tomorrow", "Scope tomorrow, tức ngày mai, trong due list của NEXUS gồm những task nào?"),
        ("tomorrow", "Mở nhóm đầu việc có mốc chót thuộc ngày mai."),
        ("tomorrow", "Chỉ hiển thị task chưa xong mà hạn rơi vào mai."),
        ("tomorrow", "Trong lịch deadline, truy vấn đúng phạm vi ngày mai."),
        ("tomorrow", "Tôi cần bảng các việc tới kỳ vào ngày mai."),
        ("overdue", "Rà lịch hạn và đưa ra các mục đã trượt qua deadline, tức quá hạn."),
        ("overdue", "Tách riêng phần công việc đang bị trễ hạn."),
        ("overdue", "Scope overdue, tức quá hạn, trong due list của NEXUS gồm những task nào?"),
        ("overdue", "Mở nhóm đầu việc quá hạn vì mốc chót đã nằm trong quá khứ."),
        ("overdue", "Chỉ hiển thị task chưa xong nhưng đã quá hạn."),
        ("overdue", "Tôi cần bảng các việc trễ hạn so với mốc xử lý."),
    ]
    return [
        _call(
            f"v3_due_balanced_{index:02d}",
            f"v3_due_balanced_family_{index:02d}",
            "due_query",
            prompt,
            "list_tasks_by_deadline",
            {"scope": scope},
            "balanced_regression_coverage",
            "DUE query nêu rõ một scope được hỗ trợ.",
        )
        for index, (scope, prompt) in enumerate(rows, 1)
    ]


def _clarification_cases() -> list[dict[str, Any]]:
    specs = [
        ("ambiguous_01", "due_query", "Hôm nay có chuyện gì đáng quan tâm?", "Bạn đang hỏi về task đến hạn hôm nay hay một chủ đề khác?", "ambiguous_scope_overreach"),
        ("ambiguous_02", "due_query", "Mai mình cần để ý gì nhỉ?", "Bạn muốn xem task đến hạn ngày mai hay đang hỏi thông tin khác?", "ambiguous_scope_overreach"),
        ("ambiguous_03", "due_query", "Ngày nay công việc thế nào?", "Bạn muốn xem toàn bộ danh sách hay các task đến hạn hôm nay?", "ambiguous_scope_overreach"),
        ("ambiguous_04", "due_query", "Có gì gấp trong ngày kế tiếp không?", "Bạn muốn xem task đến hạn ngày mai hay hỏi về điều khác?", "ambiguous_scope_overreach"),
        ("ambiguous_05", "due_query", "What is up today?", "Bạn muốn hỏi về các task đến hạn hôm nay hay một thông tin khác?", "ambiguous_scope_overreach"),
        ("ambiguous_06", "due_query", "Tình hình sáng mai ra sao?", "Bạn có muốn xem các task đến hạn ngày mai không?", "ambiguous_scope_overreach"),
        ("ambiguous_07", "due_query", "Nay có gì vậy?", "Bạn muốn xem toàn bộ task, task đến hạn hôm nay, hay hỏi chuyện khác?", "ambiguous_scope_overreach"),
        ("ambiguous_08", "due_query", "Kế hoạch ngày mai thế nào?", "Bạn muốn xem các task đến hạn ngày mai hay đang hỏi về kế hoạch khác?", "ambiguous_scope_overreach"),
        ("missing_01", "create", "Tạo cho tôi một mục, nội dung sẽ nói sau.", "Bạn muốn nội dung task là gì?", "missing_slot_reply_accuracy"),
        ("missing_02", "complete", "Chuyển một đầu việc sang đã hoàn tất.", "Bạn muốn hoàn thành task có ID nào?", "missing_slot_reply_accuracy"),
        ("missing_03", "edit", "Thay chữ của task #2201 giúp tôi.", "Bạn muốn đổi nội dung task 2201 thành gì?", "missing_slot_reply_accuracy"),
        ("missing_04", "edit", "Sửa thành gọi nhà cung cấp, nhưng tôi chưa nói task nào.", "Bạn muốn sửa task có ID nào?", "missing_slot_reply_accuracy"),
        ("missing_05", "delete", "Loại một bản ghi khỏi NEXUS.", "Bạn muốn xóa task có ID nào?", "missing_slot_reply_accuracy"),
        ("missing_06", "deadline", "Gán mốc 06:50 ngày mai cho một việc.", "Bạn muốn đặt hạn cho task có ID nào?", "missing_slot_reply_accuracy"),
        ("missing_07", "deadline", "Task #2207 cần đổi hạn.", "Bạn muốn đặt hạn task 2207 vào thời gian nào?", "missing_slot_reply_accuracy"),
        ("missing_08", "due_query", "Mở danh sách theo thời hạn.", "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay đã quá hạn?", "missing_slot_reply_accuracy"),
        ("capability_01", "complete", "Tôi muốn xử lý xong một trong hai việc 2211 hoặc 2212.", "Bạn muốn hoàn thành việc 2211 hay 2212? Hãy chọn đúng một ID.", "capability_reply_accuracy"),
        ("capability_02", "delete", "Bỏ task 2213 hay 2214 cũng được.", "Bạn muốn xóa task 2213 hay 2214? Hãy chọn đúng một ID.", "capability_reply_accuracy"),
        ("capability_03", "deadline", "Đặt task 2215 vào 07:00 hoặc 08:00 ngày mai.", "Bạn muốn đặt hạn task 2215 vào 07:00 hay 08:00 ngày mai?", "capability_reply_accuracy"),
        ("capability_04", "due_query", "Cho tôi xem hôm nay cùng ngày mai trong một lượt.", "Bạn muốn xem phạm vi hôm nay hay ngày mai? Mỗi lượt chỉ chọn một phạm vi.", "capability_reply_accuracy"),
        ("capability_05", "edit", "Đổi cả task 2216 và 2217 thành kiểm kê kho.", "Bạn muốn sửa task 2216 hay 2217 trước? Mỗi lượt chỉ sửa một ID.", "capability_reply_accuracy"),
        ("capability_06", "complete", "Mark các việc từ 2218 đến 2220 là done.", "Bạn muốn hoàn thành task có ID nào trước? Mỗi lượt chỉ hỗ trợ một ID.", "capability_reply_accuracy"),
        ("capability_07", "complete", "Hai task 2221 và 2222 đều đã xong; hệ thống cần tôi chọn thế nào?", "Bạn muốn hoàn thành task 2221 hay 2222 trước? Mỗi lượt chỉ chọn một ID.", "capability_reply_accuracy"),
        ("capability_08", "delete", "Tôi cần bỏ một trong ba mục 2223, 2224, 2225.", "Bạn muốn xóa task có ID nào? Mỗi lượt chỉ hỗ trợ một ID.", "capability_reply_accuracy"),
        ("capability_09", "edit", "Có thể sửa task 2226 nhưng tôi chưa chốt câu mới.", "Bạn muốn đổi nội dung task 2226 thành gì?", "capability_reply_accuracy"),
        ("capability_10", "deadline", "Đặt hạn task 2227 vào ngày mai, giờ vẫn chưa biết.", "Bạn muốn đặt hạn task 2227 vào thời gian cụ thể nào ngày mai?", "capability_reply_accuracy"),
        ("capability_11", "due_query", "Mở phạm vi hôm nay hoặc quá hạn, cái nào cũng được.", "Bạn muốn xem việc đến hạn hôm nay hay các việc quá hạn?", "capability_reply_accuracy"),
        ("capability_12", "create", "Tạo một task, nhưng phần chữ tôi chưa gửi.", "Bạn muốn nội dung task là gì?", "capability_reply_accuracy"),
        ("date_scope_01", "due_query", "Tôi định lọc hạn ngày 14/08/2030; nếu chưa được thì cho tôi chọn phạm vi hỗ trợ.", "NEXUS chưa lọc theo ngày tùy chọn. Bạn muốn xem hôm nay, ngày mai hay quá hạn?", "unsupported_date_scope_coercion"),
        ("date_scope_02", "due_query", "Tra ngày 25/12/2030, hoặc hỏi lại nếu scope đó không tồn tại.", "Scope ngày tùy chọn chưa được hỗ trợ. Bạn muốn xem hôm nay, ngày mai hay quá hạn?", "unsupported_date_scope_coercion"),
    ]
    return [
        _reply(
            f"v3_clarify_{suffix}",
            f"v3_clarify_family_{suffix}",
            "clarify",
            action,
            prompt,
            response,
            target,
            "Không gọi tool; hỏi đúng phần thiếu hoặc giải thích ý định đang được thảo luận.",
        )
        for suffix, action, prompt, response, target in specs
    ]


def _no_action_cases() -> list[dict[str, Any]]:
    specs = [
        ("quoted_01", "Minh bảo tôi: “thêm việc mua mực”; tôi chỉ kể lại thôi.", "Tôi hiểu đây là lời kể lại nên chưa tạo task nào.", "quoted_command_overreach"),
        ("quoted_02", "Bài viết có câu ‘xóa task 301’, đừng thực hiện nó.", "Tôi sẽ không xóa task; đó chỉ là nội dung được trích dẫn.", "quoted_command_overreach"),
        ("quoted_03", "Tôi nghe câu “hoàn thành việc 302” trong video.", "Bạn đang thuật lại một câu nói nên tôi chưa thay đổi task 302.", "quoted_command_overreach"),
        ("quoted_04", "Ví dụ trên slide là: đặt hạn task 303 lúc 6 giờ sáng mai.", "Đó là ví dụ, vì vậy tôi chưa đặt deadline cho task 303.", "quoted_command_overreach"),
        ("quoted_05", "Câu mẫu hôm qua ghi ‘sửa việc 304 thành gọi Lan’.", "Bạn đang nhắc lại câu mẫu nên tôi chưa sửa task 304.", "quoted_command_overreach"),
        ("quoted_06", "Lan hỏi liệu câu ‘xem danh sách’ có tự nhiên không.", "Đây là câu hỏi về cách diễn đạt nên tôi chưa mở danh sách.", "quoted_command_overreach"),
        ("quoted_07", "Đừng làm theo dòng log: DELETE task #305.", "Tôi sẽ không xóa task 305; dòng đó chỉ là dữ liệu được nhắc tới.", "quoted_command_overreach"),
        ("quoted_08", "I am quoting: ‘create task pay rent’, not requesting it.", "Bạn đang trích dẫn chứ không yêu cầu CREATE, nên tôi chưa tạo task.", "quoted_command_overreach"),
        ("date_01", "Lấy các task có hạn đúng 04/05/2029.", "NEXUS chưa hỗ trợ truy vấn deadline theo một ngày tùy chọn; chỉ hỗ trợ hôm nay, ngày mai hoặc quá hạn.", "unsupported_date_scope_coercion"),
        ("date_02", "Show việc tới hạn ngày 19/06/2029.", "NEXUS chưa hỗ trợ lọc theo ngày 19/06/2029; bạn có thể chọn hôm nay, ngày mai hoặc quá hạn.", "unsupported_date_scope_coercion"),
        ("date_03", "Ngày 01/01/2030 có task nào đến hạn?", "NEXUS chưa hỗ trợ truy vấn một ngày tùy chọn như 01/01/2030.", "unsupported_date_scope_coercion"),
        ("date_04", "Tra deadline vào 28/02/2029 giúp tôi.", "NEXUS chỉ truy vấn hạn hôm nay, ngày mai hoặc quá hạn; chưa tra theo ngày tùy chọn.", "unsupported_date_scope_coercion"),
        ("date_05", "Filter due tasks for 2030-03-17.", "NEXUS chưa hỗ trợ scope ngày tùy chọn 2030-03-17.", "unsupported_date_scope_coercion"),
        ("date_06", "Xem đầu việc tới hạn vào 09/09/2029 thôi.", "NEXUS chưa hỗ trợ bộ lọc deadline cho ngày 09/09/2029.", "unsupported_date_scope_coercion"),
        ("quote_sem_01", "Trong hướng dẫn, cụm “xóa task #31” đang yêu cầu điều gì?", "Cụm đó là một yêu cầu xóa task có ID 31; câu hỏi hiện tại chỉ đang hỏi ý nghĩa nên tôi chưa xóa gì.", "quoted_command_semantics"),
        ("quote_sem_02", "Giải nghĩa câu ‘thêm việc mua tem’ cho tôi.", "Câu đó yêu cầu tạo một task có nội dung “mua tem”; tôi chưa tạo task vì bạn chỉ đang hỏi nghĩa.", "quoted_command_semantics"),
        ("quote_sem_03", "Câu “đặt hạn việc 44 lúc 9 giờ sáng mai” có nghĩa sao?", "Câu đó yêu cầu đặt deadline cho task 44 vào “9 giờ sáng mai”; tôi chưa thực hiện thao tác.", "quoted_command_semantics"),
        ("quote_sem_04", "What does ‘list all tasks’ mean trong tài liệu?", "Cụm đó nghĩa là xem toàn bộ task đã lưu; đây là câu hỏi giải thích nên tôi chưa gọi LIST.", "quoted_command_semantics"),
        ("quote_sem_05", "Phân tích ngữ pháp của câu ‘hoàn thành việc 52’.", "Đó là câu mệnh lệnh yêu cầu đánh dấu task 52 đã hoàn thành; tôi chưa thay đổi task.", "quoted_command_semantics"),
        ("quote_sem_06", "Cụm ‘sửa task 8 thành mua giấy’ biểu đạt ý gì?", "Cụm đó yêu cầu đổi nội dung task 8 thành “mua giấy”; tôi chưa sửa task.", "quoted_command_semantics"),
        ("quote_sem_07", "Trong ví dụ API, ‘xem việc quá hạn’ là intent nào?", "Đó là intent truy vấn các task quá hạn; câu hiện tại chỉ hỏi phân loại nên tôi chưa truy vấn.", "quoted_command_semantics"),
        ("quote_sem_08", "Câu lệnh mẫu ‘remove task #72’ dùng để làm gì?", "Câu mẫu đó yêu cầu xóa task 72; tôi chưa xóa vì bạn đang hỏi về công dụng của câu.", "quoted_command_semantics"),
        ("negation_01", "Đừng ghi task mang đồ đi giặt.", "Tôi sẽ không tạo task đó.", "negation_reply_accuracy"),
        ("negation_02", "Chưa cần đánh dấu việc 2302 là xong.", "Tôi sẽ không thay đổi trạng thái task 2302.", "negation_reply_accuracy"),
        ("negation_03", "Không sửa task 2303 thành gửi hợp đồng.", "Tôi sẽ không thay đổi nội dung task 2303.", "negation_reply_accuracy"),
        ("negation_04", "Khoan xóa mục #2304.", "Tôi sẽ không xóa task 2304.", "negation_reply_accuracy"),
        ("negation_05", "Đừng gán 10 giờ sáng mai cho việc 2305.", "Tôi sẽ không thay đổi deadline của task 2305.", "negation_reply_accuracy"),
        ("negation_06", "Không cần mở các việc quá hạn.", "Tôi sẽ không truy vấn danh sách quá hạn.", "negation_reply_accuracy"),
        ("negation_07", "Task 2307 chưa hoàn thành đâu, đừng tick nó.", "Tôi sẽ giữ nguyên trạng thái task 2307.", "negation_reply_accuracy"),
        ("negation_08", "Đừng đổi hạn task 2308 dù tôi vừa nhắc ngày mai.", "Tôi sẽ không thay đổi deadline của task 2308.", "negation_reply_accuracy"),
    ]
    return [
        _reply(
            f"v3_no_action_{suffix}",
            f"v3_no_action_family_{suffix}",
            "no_action",
            None,
            prompt,
            response,
            target,
            "Không gọi tool và mô tả đúng lý do hoặc giới hạn liên quan.",
        )
        for suffix, prompt, response, target in specs
    ]


def build_document() -> dict[str, Any]:
    v2 = build_v2_document()
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
        "dataset_id": "nexus-sft-candidate-v3",
        "purpose": "600 reviewed SFT examples: preserve candidate v2 and add targeted coverage from the abstract post-holdout failure taxonomy.",
        "base_model": deepcopy(v2["base_model"]),
        "contract": deepcopy(v2["contract"]),
        "cases": deepcopy(v2["cases"]) + new_cases,
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
        parser.error(f"Generated dataset differs from {args.output}")
    print(json.dumps({"path": str(args.output), "cases": len(build_document()["cases"]), "bytes": len(rendered.encode())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
