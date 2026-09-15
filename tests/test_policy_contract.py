"""Behavior regressions from scoring v2 and the single-call contract."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from nexus.agent.policy import PolicyResult, policy_for_tool
from scripts.eval_qwen import evaluate_case

DATASET = Path(__file__).resolve().parents[1] / 'evals/current_scope_tasks_v2.json'


def proposal_response(calls):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'content': None,
        'tool_calls': [{'id': f'call-{i}', 'type': 'function', 'function': {
            'name': call['name'], 'arguments': json.dumps(call['arguments'], ensure_ascii=False)
        }} for i, call in enumerate(calls)]
    }}]}


class PolicyContractTests(unittest.TestCase):
    def test_four_false_rejections_through_real_runtime(self):
        cases = json.loads(DATASET.read_text())
        for case in cases:
            if case['id'] not in {'create_one', 'create_colloquial', 'create_two_lines', 'list_empty'}:
                continue
            with self.subTest(case=case['id']):
                response = proposal_response(case['expected_calls'])
                result = evaluate_case(case, lambda _: response)
                self.assertEqual(result['model_proposal_status'], 'pass')
                self.assertEqual(result['system_action_status'], 'pass')
                self.assertFalse(result['policy']['false_rejection'])

    def test_create_framing_variants(self):
        cases = [
            ('Thêm việc tưới cây vào danh sách giúp mình.', 'tưới cây'),
            ('Note giúp tui: kiểm tra pin', 'kiểm tra pin'),
            ('Note giúp tôi: gửi email', 'gửi email'),
            ('Ghi giúp mình việc gọi cho bố nhé.', 'gọi cho bố'),
            ('Ghi lại ba việc sau, mỗi dòng một việc:\na\nb\nc', 'a\nb\nc'),
            ('Ghi lại 12 việc sau, mỗi dòng một việc:\na\nb', 'a\nb'),
            ('Ghi lại các việc sau:\na\nb', 'a\nb'),
            ('Thêm việc mua sữa.', 'mua sữa'),
            ('Thêm việc mua sữa.', 'mua sữa.'),
            ('Thêm việc:\n  sửa xe.  ', 'sửa xe.'),
            ('Thêm việc\nsửa xe.', 'sửa xe.'),
            ('Thêm việc Ôn C++: vector', 'Ôn C++: vector'),
            ('Thêm việc đi ăn lúc 8:00', 'đi ăn lúc 8:00'),
            ('Thêm việc: mua thuốc không cần kê đơn.', 'mua thuốc không cần kê đơn.'),
            ('Thêm việc: đừng quên gọi mẹ.', 'đừng quên gọi mẹ.'),
            ('Thêm việc: sửa task mua sữa.', 'sửa task mua sữa.'),
            ('Thêm việc: đọc ví dụ về xóa task.', 'đọc ví dụ về xóa task.'),
            ('Thêm việc: giúp tôi.', 'giúp tôi.'),
            ('ghi lại việc ghi lại việc nhé.', 'ghi lại việc'),
            ('Thêm các việc:\r\na\r\nb', 'a\r\nb'),
            ('Thêm các việc:\na\n\na', 'a\n\na'),
        ]
        for prompt, content in cases:
            with self.subTest(prompt=prompt, content=content):
                args = {'content': content}
                original = deepcopy(args)
                self.assertEqual(policy_for_tool(prompt, 'create_task', args).result, PolicyResult.ALLOW)
                self.assertEqual(args, original)

    def test_sau_inside_task_is_not_a_multiline_introduction(self):
        prompt = 'Thêm việc sau giờ làm'
        self.assertEqual(policy_for_tool(prompt, 'create_task', {'content': 'sau giờ làm'}).result, PolicyResult.ALLOW)
        self.assertEqual(policy_for_tool(prompt, 'create_task', {'content': 'giờ làm'}).reason.value, 'content_boundary_mismatch')

    def test_substrings_must_cover_the_complete_content(self):
        cases = [
            ('Thêm việc mua sữa và gọi mẹ', 'mua sữa'),
            ('Thêm việc mua sữa', 'mua'),
            ('Thêm việc mua sữa', 'mua sữ'),
            ('Thêm các việc:\nmua sữa\ngọi mẹ', 'mua sữa'),
            ('Thêm việc: sửa xe.', 'sửa xe'),
            ('Thêm việc: mua sữa nhé.', 'mua sữa'),
            ('Thêm việc ghi lại việc nhé.', 'việc'),
            ('Thêm việc: a và a', 'a'),
            ('Thêm các việc:\na\na', 'a'),
        ]
        for prompt, content in cases:
            with self.subTest(prompt=prompt, content=content):
                decision = policy_for_tool(prompt, 'create_task', {'content': content})
                self.assertEqual(decision.result, PolicyResult.REJECT)
                self.assertEqual(decision.reason.value, 'content_boundary_mismatch')

    def test_rewrites_remain_not_grounded(self):
        cases = [
            ('Thêm việc: mua sữa', 'Mua sữa'),
            ('Thêm việc: mua sữa', 'mua sữa lúc 8h'),
            ('Thêm các việc:\na\nb', 'b\na'),
            ('Thêm việc: a, b', 'a\nb'),
            ('Thêm việc: a và b', 'a\nb'),
            ('Ghi giúp mình việc gọi cho mẹ nhé.', 'Gọi mẹ để hỏi về cuộc sống gia đình.'),
        ]
        for prompt, content in cases:
            with self.subTest(prompt=prompt):
                decision = policy_for_tool(prompt, 'create_task', {'content': content})
                self.assertEqual(decision.result, PolicyResult.REJECT)
                self.assertEqual(decision.reason.value, 'content_not_grounded')

    def test_no_authorization_from_quotes_negation_or_other_actions(self):
        for prompt in [
            'Mua sữa.', 'Đừng thêm việc mua sữa.', 'Không cần ghi việc mua sữa.',
            'Xóa task mua sữa.', 'Sửa task mua sữa.',
            'Tôi lấy ví dụ: thêm việc mua sữa.',
            '“Thêm việc mua sữa” là câu ví dụ.',
            'Thêm việc mua sữa chỉ là ví dụ, đừng thực hiện.',
            'Thêm việc mua sữa, nhưng đừng lưu.',
        ]:
            with self.subTest(prompt=prompt):
                self.assertEqual(policy_for_tool(prompt, 'create_task', {'content': 'mua sữa'}).result, PolicyResult.REJECT)

    def test_list_questions_are_about_saved_tasks(self):
        for prompt in ['Danh sách việc của tôi hiện có gì?', 'Danh sách của tui có gì?',
                       'Todo của mình hiện có gì?', 'Mình đã ghi những việc gì rồi nhỉ?',
                       'Hiển thị danh sách task']:
            with self.subTest(prompt=prompt):
                self.assertEqual(policy_for_tool(prompt, 'list_tasks', {}).result, PolicyResult.ALLOW)
        for prompt in ['Bữa sáng có gì?', 'Đây là danh sách của tôi.',
                       'Ví dụ: xem danh sách', '“Xem danh sách” có nghĩa là gì?',
                       'Đừng xem danh sách.', 'Xóa danh sách.',
                       'Hôm qua tôi xem danh sách.', 'Tôi thích xem danh sách.']:
            with self.subTest(prompt=prompt):
                self.assertEqual(policy_for_tool(prompt, 'list_tasks', {}).result, PolicyResult.REJECT)

    def test_one_multiline_call_succeeds_but_two_calls_do_not_write(self):
        case = next(c for c in json.loads(DATASET.read_text()) if c['id'] == 'create_two_lines')
        one_call = proposal_response(case['expected_calls'])
        result = evaluate_case(case, lambda _: one_call)
        self.assertEqual(result['system_action_status'], 'pass')
        two_calls = proposal_response([
            {'name': 'create_task', 'arguments': {'content': 'mua sữa'}},
            {'name': 'create_task', 'arguments': {'content': 'gọi mẹ'}},
        ])
        result = evaluate_case(case, lambda _: two_calls)
        self.assertEqual(result['model_proposal_status'], 'fail')
        self.assertEqual(result['executed_calls'], [])
        self.assertEqual(result['database_after'], [])
        self.assertEqual(len(result['rejected_calls']), 2)

    def test_rejected_payload_does_not_mutate_database_or_arguments(self):
        case = {'id': 'partial', 'prompt': 'Thêm việc: mua sữa và gọi mẹ',
                'initial_tasks': ['giữ nguyên'],
                'expected_calls': [{'name': 'create_task', 'arguments': {'content': 'mua sữa và gọi mẹ'}}],
                'expected_tasks': ['giữ nguyên', 'mua sữa và gọi mẹ'], 'reply_expectation': 'Xác nhận.'}
        response = proposal_response([{'name': 'create_task', 'arguments': {'content': 'mua sữa'}}])
        original = deepcopy(response)
        result = evaluate_case(case, lambda _: response)
        self.assertEqual(response, original)
        self.assertEqual(result['database_before'], result['database_after'])
        self.assertEqual(result['executed_calls'], [])

    def test_dataset_v2_only_changes_multicall_option(self):
        old = json.loads(DATASET.with_name('current_scope_tasks.json').read_text())
        new = json.loads(DATASET.read_text())
        self.assertEqual(len(new), 20)
        for before, after in zip(old, new):
            before.pop('expected_call_options', None)
            options = after.pop('expected_call_options', [after['expected_calls']])
            self.assertEqual(options, [after['expected_calls']])
            self.assertEqual(before, after)
