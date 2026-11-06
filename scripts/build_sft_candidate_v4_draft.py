"""Build a small, reviewable SFT amendment after the framing diagnostic."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_sft_candidate_v3 import _call, _reply, build_document as build_v3_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals/sft_candidate_v4_draft.json"


def _calls(
    action: str, tool: str, target: str,
    specs: list[tuple[str, str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        _call(
            f"v4_{action}_{index:02d}", f"v4_{action}_family_{index:02d}",
            action, prompt, tool, arguments, target,
            f"Yêu cầu hành động rõ ràng; tool và arguments phải khớp lời người dùng. Trục kiểm tra: {focus}.",
        )
        for index, (focus, prompt, arguments) in enumerate(specs, 1)
    ]


def _target_calls() -> list[dict[str, Any]]:
    deadline = [
        ("deadline_frame", "Cho tôi đặt hạn việc 274: 6 giờ 10 phút sáng mai.", {"id": 274, "when": "6 giờ 10 phút sáng mai"}),
        ("deadline_frame", "Đặt hạn việc 832 vào 19:25 hôm nay.", {"id": 832, "when": "19:25 hôm nay"}),
        ("deadline_frame", "Đặt hạn cho task 406 lúc 8 giờ tối ngày mai.", {"id": 406, "when": "8 giờ tối ngày mai"}),
        ("deadline_frame", "Chốt deadline của việc 193 ở 17:40 ngày 28/11/2028.", {"id": 193, "when": "17:40 ngày 28/11/2028"}),
        ("deadline_frame", "Việc 629: đặt hạn 10 giờ sáng hôm nay.", {"id": 629, "when": "10 giờ sáng hôm nay"}),
        ("deadline_frame", "Hạn của việc 702, hãy chuyển sang 07:30 ngày 29/11/2028.", {"id": 702, "when": "07:30 ngày 29/11/2028"}),
        ("deadline_frame", "Set deadline cho task 975 vào 9 giờ 20 phút tối mai.", {"id": 975, "when": "9 giờ 20 phút tối mai"}),
        ("deadline_frame", "Với việc ID 461, hạn mới là 14:05 ngày mai; ghi lại.", {"id": 461, "when": "14:05 ngày mai"}),
        ("deadline_frame", "Đặt hạn việc 358 ở 8 giờ sáng mai, đúng ID đó.", {"id": 358, "when": "8 giờ sáng mai"}),
        ("deadline_frame", "Xin cập nhật deadline cho việc 847 thành 21:10 hôm nay.", {"id": 847, "when": "21:10 hôm nay"}),
    ]
    list_all = [
        ("list_completed", "Cho tôi xem danh sách, kể cả các việc đã hoàn thành.", {}),
        ("list_completed", "Liệt kê tất cả task đã lưu; các mục done cũng phải hiện.", {}),
        ("list_completed", "Xem lại mọi đầu việc, gồm cả những việc đã xong.", {}),
        ("list_completed", "Đọc danh sách NEXUS đầy đủ, cả các task completed.", {}),
        ("list_completed", "Mở toàn bộ danh sách việc đã ghi, không lọc theo trạng thái.", {}),
        ("list_completed", "Show tất cả task trong sổ, cả hoàn thành và chưa hoàn thành.", {}),
        ("list_completed", "Tôi cần xem cả việc chưa làm lẫn việc đã làm xong.", {}),
        ("list_completed", "Đưa ra tất cả các đầu việc hiện có, đã xong cũng vậy.", {}),
        ("list_completed", "Liệt kê mọi task, bao gồm các mục hoàn thành rồi.", {}),
        ("list_completed", "Xem các việc đã lưu bất kể trạng thái hoàn thành.", {}),
    ]
    return (
        _calls("deadline", "set_task_deadline", "deadline_framing_coverage", deadline)
        + _calls("list", "list_tasks", "list_completed_contract", list_all)
    )


def _controls() -> list[dict[str, Any]]:
    specs = [
        ("clarify", "deadline", "Đặt hạn lúc 6 giờ sáng mai cho một việc, ID tôi sẽ nói sau.", "Bạn muốn đặt hạn cho việc có ID nào?"),
        ("clarify", "deadline", "Cập nhật hạn việc 618, thời điểm tôi sẽ gửi sau.", "Bạn muốn đặt hạn việc 618 vào thời gian nào?"),
        ("clarify", "deadline", "Đặt hạn cho việc 619 và 620 lúc 7 giờ tối mai; tôi sẽ chọn một việc trước.", "Bạn muốn đặt hạn cho việc 619 hay 620?"),
        ("clarify", "due_query", "Cho tôi xem việc theo hạn, phạm vi tôi sẽ nói sau.", "Bạn muốn xem việc đến hạn hôm nay, ngày mai hay quá hạn?"),
        ("clarify", "edit", "Tôi muốn sửa một task thành đặt giấy A4; ID sẽ gửi sau.", "Bạn muốn sửa task có ID nào?"),
        ("no_action", None, "Câu ví dụ là ‘đặt hạn việc 621 lúc 8 giờ sáng mai’; hãy giải thích ý nghĩa thôi.", "Đó là câu yêu cầu đặt hạn; tôi chưa thực hiện thao tác nào."),
        ("no_action", None, "Đừng đặt hạn việc 622 vào 19:20 hôm nay.", "Tôi sẽ không thay đổi deadline của việc 622."),
        ("no_action", None, "Trong tài liệu có câu ‘xem danh sách kể cả mục hoàn thành’; tôi chỉ đang trích dẫn.", "Bạn đang trích dẫn nên tôi chưa mở danh sách."),
        ("no_action", None, "Không cần hiển thị danh sách, dù có việc đã hoàn thành.", "Tôi sẽ không mở danh sách."),
        ("no_action", None, "Chỉ hiện riêng những task đã hoàn thành, đừng hiện task chưa xong.", "NEXUS chưa hỗ trợ lọc riêng task đã hoàn thành; bạn có thể xem toàn bộ danh sách."),
    ]
    return [
        _reply(
            f"v4_control_{index:02d}", f"v4_control_family_{index:02d}",
            behavior, action, prompt, response, "boundary_control",
            "Không gọi tool khi thiếu dữ liệu, bị phủ định, đang trích dẫn hoặc yêu cầu bộ lọc chưa hỗ trợ.",
        )
        for index, (behavior, action, prompt, response) in enumerate(specs, 1)
    ]


def _other_action_calls() -> list[dict[str, Any]]:
    groups = [
        ("create", "create_task", [
            ("regression", "Thêm việc: thay bóng đèn bàn", {"content": "thay bóng đèn bàn"}),
            ("regression", "Lưu một task mới có nội dung: đọc tài liệu cổng USB-C", {"content": "đọc tài liệu cổng USB-C"}),
        ]),
        ("complete", "complete_task", [
            ("regression", "Đánh dấu việc 113 đã hoàn thành.", {"id": 113}),
            ("regression", "Tôi làm xong task 864 rồi; cập nhật trạng thái.", {"id": 864}),
        ]),
        ("edit", "update_task", [
            ("regression", "Sửa việc 371 thành mua dây sạc mới.", {"id": 371, "content": "mua dây sạc mới"}),
            ("regression", "Đổi nội dung task 972 thành: gửi bản thiết kế B2", {"id": 972, "content": "gửi bản thiết kế B2"}),
        ]),
        ("delete", "delete_task", [
            ("regression", "Xóa việc 487 khỏi danh sách.", {"id": 487}),
            ("regression", "Bỏ hẳn task 931; ứng dụng hãy hỏi xác nhận.", {"id": 931}),
        ]),
        ("due_query", "list_tasks_by_deadline", [
            ("regression", "Xem các việc đến hạn hôm nay.", {"scope": "today"}),
            ("regression", "Cho tôi biết task nào đã quá hạn.", {"scope": "overdue"}),
        ]),
    ]
    return [case for action, tool, specs in groups
            for case in _calls(action, tool, "other_action_regression", specs)]


def build_document() -> dict[str, Any]:
    v3 = build_v3_document()
    additions = _target_calls() + _controls() + _other_action_calls()
    return {
        "schema_version": 1,
        "dataset_id": "nexus-sft-candidate-v4-draft",
        "purpose": "Minimal framing amendment with no-tool controls and other-action coverage; unreviewed training draft.",
        "base_model": deepcopy(v3["base_model"]),
        "contract": deepcopy(v3["contract"]),
        "source_v3_sha256": hashlib.sha256((ROOT / "evals/sft_candidate_v3.json").read_bytes()).hexdigest(),
        "cases": deepcopy(v3["cases"]) + additions,
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
        parser.error("Draft dataset differs from its builder")
    print(f"Candidate v4 draft: {len(build_document()['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
