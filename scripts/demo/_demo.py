"""Human-readable demo output; all writes are confined to the supplied demo directory."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nexus.agent.client import (
    DEFAULT_MODEL_ID,
    ENDPOINT,
    PostToolExecutionError,
    run_turn,
)
from nexus.agent.policy import policy_for_tool
from nexus.agent.session import AgentSession
from nexus.storage.sqlite_db import initialize_database, list_tasks
from scripts.eval_qwen import send_chat

ROOT = Path(__file__).resolve().parents[2]


def dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def policy_demo():
    examples = [
        ('Thêm việc mua sữa vào danh sách giúp tôi.', 'create_task', {'content': 'mua sữa'}),
        ('Note giúp tui: mua pin chuột', 'create_task', {'content': 'mua pin chuột'}),
        ('Ghi lại hai việc sau, mỗi dòng một việc:\nmua sữa\ngọi mẹ',
         'create_task', {'content': 'mua sữa\ngọi mẹ'}),
        ('Danh sách việc của tôi hiện có gì?', 'list_tasks', {}),
        ('Thêm việc mua sữa và gọi mẹ', 'create_task', {'content': 'mua sữa'}),
        ('Thêm việc: mua sữa', 'create_task', {'content': 'Mua sữa'}),
        ('Thêm việc: sửa xe.', 'create_task', {'content': 'sửa xe'}),
        ('Thêm việc: đừng quên gọi mẹ.', 'create_task', {'content': 'đừng quên gọi mẹ.'}),
        ('Mua sữa.', 'create_task', {'content': 'Mua sữa.'}),
        ('Đừng thêm việc mua sữa.', 'create_task', {'content': 'mua sữa'}),
        ('Thêm việc:', 'create_task', {'content': ''}),
    ]
    print('Các proposal dưới đây do demo cung cấp, không phải output của Qwen.')
    for i, (prompt, name, arguments) in enumerate(examples, 1):
        print(f'\n[{i}] Người dùng: {prompt}')
        print(f'Proposal: {name}({json.dumps(arguments, ensure_ascii=False)})')
        decision = policy_for_tool(prompt, name, arguments)
        print(f'Policy: {decision.result.value} — {decision.reason.value}')


def fake_response(calls):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'content': None,
        'tool_calls': [{'id': f'demo-{i}', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False),
        }} for i, (name, arguments) in enumerate(calls)],
    }}]}


def interactive_examples():
    print('Nhập từng yêu cầu một dòng; /exit hoặc Ctrl+D để kết thúc.', flush=True)
    while True:
        try:
            prompt = input('NEXUS > ')
        except EOFError:
            print()
            return
        if prompt.strip().lower() == '/exit':
            return
        if prompt.strip():
            yield prompt, None


def agent_demo(directory, live, prompt, interactive=False):
    database = directory / 'tasks.db'
    initialize_database(database)
    settings = {'model': DEFAULT_MODEL_ID, 'temperature': 0,
                'top_p': 0.8, 'top_k': 20, 'min_p': 0, 'max_tokens': 256,
                'chat_template_kwargs': {'enable_thinking': False}}
    if live:
        print(f'Endpoint: {ENDPOINT}\nPrompt: v1; temperature: 0')
        if interactive:
            examples = interactive_examples()
        else:
            inputs = [prompt] if prompt is not None else [
                'Thêm việc: mua sữa', 'Xem danh sách', 'Mua bánh.',
            ]
            examples = [(text, None) for text in inputs]
    else:
        examples = [
            ('Thêm việc mua sữa vào danh sách giúp tôi.',
             [('create_task', {'content': 'mua sữa'})]),
            ('Thêm các việc:\nđọc sách\ngọi mẹ',
             [('create_task', {'content': 'đọc sách\ngọi mẹ'})]),
            ('Thêm việc mua sữa và tưới cây',
             [('create_task', {'content': 'mua sữa'})]),
            ('Thêm việc: nấu cơm', [('create_task', {'content': 'Nấu cơm'})]),
            ('Mua bánh.', [('create_task', {'content': 'Mua bánh.'})]),
            ('Thêm các việc:\na\nb',
             [('create_task', {'content': 'a'}), ('create_task', {'content': 'b'})]),
            ('Xem danh sách', [('list_tasks', {})]),
        ]
    turns = []
    for text, calls in examples:
        print(f'\n=== Người dùng: {text} ===', flush=True)
        before = [asdict(task) for task in list_tasks(database)]
        record = {'prompt': text, 'database_before': before, 'error': None}
        print('0. SQLite trước lượt:')
        dump(before)

        def generate(payload):
            record['model_request'] = payload
            print('1. Gửi tới Qwen:' if live else '1. Request cho model giả lập:',
                  flush=True)
            dump({'prompt_version': 'v1', 'temperature': settings['temperature'],
                  'tools': [tool['function']['name'] for tool in payload['tools']],
                  'user_message': payload['messages'][-1]['content']})
            response = send_chat(ENDPOINT, payload) if live else fake_response(calls)
            record['model_response'] = response
            choice = response['choices'][0]
            message = choice['message']
            print('2. Qwen trả về (chưa được thực thi):' if live
                  else '2. Response giả lập (chưa được thực thi):', flush=True)
            dump({'finish_reason': choice.get('finish_reason'),
                  'content': message.get('content'),
                  'tool_calls': message.get('tool_calls', [])})
            return response

        try:
            turn = run_turn(database, text, generate, prompt_version='v1', settings=settings)
            record.update(turn)
            print('3. Proposal đã giải mã:')
            dump(turn['proposed_calls'])
            print('4. Policy (cho phép / từ chối và reason):')
            dump({'authorized_calls': turn['authorized_calls'],
                  'rejected_calls': turn['rejected_calls']})
            print('5. Tool đã thực thi:')
            dump(turn['calls'])
            print(f"6. Trạng thái: {turn['status']}; phản hồi ứng dụng: {turn['reply']}",
                  flush=True)
        except (RuntimeError, ValueError, OSError) as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            if isinstance(error, PostToolExecutionError):
                record['calls'] = error.executed_calls
                print('5. Tool đã hoàn tất trước lỗi:')
                dump(error.executed_calls)
            print(f'Lỗi: {error}', flush=True)
            if live:
                print('Nếu chưa chạy server, mở terminal khác và chạy scripts/start_qwen.sh.')
        record['database_after'] = [asdict(task) for task in list_tasks(database)]
        print('7. SQLite sau lượt:')
        dump(record['database_after'])
        turns.append(record)
        (directory / 'trace.json').write_text(json.dumps(turns, ensure_ascii=False, indent=2)+'\n')
        if record['error']:
            return 1
    (directory / 'trace.json').write_text(json.dumps(turns, ensure_ascii=False, indent=2)+'\n')
    print(f'\nTrace: {directory / "trace.json"}\nSQLite: {database}')
    return 0


def session_demo(directory):
    database = directory / 'tasks.db'
    initialize_database(database)
    session = AgentSession(database, settings={'model': 'demo', 'temperature': 0})
    examples = [
        'Thêm việc',
        'mua sữa\ngọi mẹ',
        'Thêm việc',
        'thôi',
        'Xem danh sách',
    ]
    turns = []

    def generate(payload):
        print('Model giả lập được gọi cho lệnh LIST.', flush=True)
        return fake_response([('list_tasks', {})])

    for prompt in examples:
        before = [asdict(task) for task in list_tasks(database)]
        pending_before = session.pending_create
        print(f'\n=== Người dùng: {prompt} ===', flush=True)
        print(f'Chờ nội dung CREATE trước lượt: {pending_before}')
        print('SQLite trước lượt:')
        dump(before)
        result = session.run_turn(prompt, generate)
        after = [asdict(task) for task in list_tasks(database)]
        print(f"Nguồn: {result['source']}; trạng thái: {result['status']}")
        print('Proposal / quyền thực thi / tool đã chạy:')
        dump({'proposed_calls': result['proposed_calls'],
              'authorized_calls': result['authorized_calls'],
              'rejected_calls': result['rejected_calls'],
              'calls': result['calls']})
        print(f"Phản hồi: {result['reply']}")
        print(f'Chờ nội dung CREATE sau lượt: {session.pending_create}')
        print('SQLite sau lượt:')
        dump(after)
        turns.append({'prompt': prompt, 'pending_before': pending_before,
                      'database_before': before, **result,
                      'pending_after': session.pending_create,
                      'database_after': after})

    trace_path = directory / 'trace.json'
    trace_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2) + '\n')
    print(f'\nTrace: {trace_path}\nSQLite: {database}')
    return 0


def read_report(path):
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or not isinstance(report.get('cases'), list) or 'summary' not in report:
        raise ValueError(f'{path} không phải báo cáo evaluator (cần cases và summary).')
    return report


def report_demo(path):
    if path is None:
        result_dir = ROOT / 'evals/results'
        candidates = list(result_dir.glob('*.json')) + list(result_dir.glob('**/report.json'))
        for candidate in sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                read_report(candidate)
            except (ValueError, OSError):
                continue
            path = candidate
            break
        if path is None:
            raise ValueError('Chưa có báo cáo. Chạy scripts/demo/05_benchmark.sh trước.')
    report = read_report(path)
    summary = report['summary']
    print(f'File: {path.resolve()}')
    print(f"Prompt: {report.get('prompt_version', '?')}; số ca: {summary.get('total', '?')}")
    for key in ('model_proposal', 'system_action', 'system_end_to_end', 'reply_hygiene', 'policy', 'safety'):
        print(f'{key}: {json.dumps(summary.get(key), ensure_ascii=False)}')
    print('\nChi tiết từng ca:')
    for case in report['cases']:
        print(f"\n{case['id']}: model={case.get('model_proposal_status', '?')}, "
              f"system={case.get('system_action_status', '?')}, "
              f"reply={case.get('reply_hygiene_status', '?')}, "
              f"system_e2e={case.get('system_end_to_end_status', '?')}")
        for call in case.get('rejected_calls', []):
            print(f"  Chặn {call.get('name')}: {call.get('reason')}")
        if case.get('error'):
            print(f"  Lỗi: {case['error']}")
        print(f"  AI: {case.get('reply')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('policy')
    for command in ('offline', 'live', 'session'):
        sub = commands.add_parser(command)
        sub.add_argument('--directory', type=Path, required=True)
        sub.add_argument('--prompt')
        if command == 'live':
            sub.add_argument('--interactive', action='store_true')
    commands.add_parser('report').add_argument('path', type=Path, nargs='?')
    args = parser.parse_args()
    try:
        if args.command == 'policy':
            policy_demo()
        elif args.command == 'report':
            report_demo(args.path)
        elif args.command == 'session':
            return session_demo(args.directory)
        else:
            if args.command == 'live' and args.interactive and args.prompt is not None:
                parser.error('--interactive không dùng cùng --prompt')
            return agent_demo(args.directory, args.command == 'live', args.prompt,
                              getattr(args, 'interactive', False))
    except (ValueError, OSError) as error:
        print(f'Lỗi: {error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
