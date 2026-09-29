"""Create and validate manual reply reviews bound to an immutable proposal report."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.eval_proposals import strict_json

CRITERIA = ('asks_missing', 'avoids_reasking_known', 'no_premature_success', 'examples_supported')
VALUES = {'pending', 'pass', 'fail', 'not_applicable'}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_report(path):
    source = Path(path).read_bytes()
    report = strict_json(source)
    if not isinstance(report, dict) or report.get('scoring_version') != 'proposal-v1':
        raise ValueError('Expected a proposal-v1 report')
    cases = report.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Report needs at least one case')
    seen = set()
    for c in cases:
        if not isinstance(c, dict) or not isinstance(c.get('id'), str) or not c['id'].strip() or c['id'] in seen:
            raise ValueError('Report case IDs must be non-empty and unique')
        seen.add(c['id'])
        if not isinstance(c.get('expected'), dict) or c['expected'].get('behavior') not in ('call', 'clarify', 'no_action'):
            raise ValueError('Invalid expected behavior in report')
        if not isinstance(c.get('prompt'), str) or not isinstance(c.get('known_fields', {}), dict):
            raise ValueError('Invalid prompt or known_fields')
        if c.get('automatic_status') not in ('pass', 'fail', 'error') or 'raw_response' not in c:
            raise ValueError('Invalid automatic status or missing raw response')
        if c['automatic_status'] != 'error' and c['raw_response'] is None:
            raise ValueError('Evaluated case must have a raw response')
        if c['automatic_status'] == 'error' and c['raw_response'] is not None:
            raise ValueError('Generation error cannot carry a scored response')
    return report, hashlib.sha256(source).hexdigest()


def template(path):
    report, digest = read_report(path)
    rows = []
    for c in report['cases']:
        reviewable = c['automatic_status'] != 'error'
        asks = reviewable and c['expected']['behavior'] == 'clarify'
        checks = {key: {'result': 'pending' if reviewable else 'not_applicable', 'evidence': ''} for key in CRITERIA}
        if not asks:
            checks['asks_missing']['result'] = 'not_applicable'
        rows.append({
            'id': c['id'], 'response_sha256': digest_json(c['raw_response']),
            'context_sha256': digest_json({'prompt': c['prompt'], 'expected': c['expected'],
                                          'known_fields': c.get('known_fields', {})}),
            'automatic_status': c['automatic_status'], 'reviewable': reviewable,
            'prompt': c['prompt'], 'expected': deepcopy(c['expected']),
            'known_fields': deepcopy(c.get('known_fields', {})),
            'reply': c.get('model_reply'), 'criteria': checks, 'notes': '',
        })
    return {'schema_version': 1, 'report_sha256': digest,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'reviewer': {'name': '', 'kind': ''}, 'cases': rows}


def validate_review(report_path, review):
    base = template(report_path)
    if not isinstance(review, dict) or review.get('schema_version') != 1:
        raise ValueError('Expected a review schema_version=1')
    if set(review) != set(base):
        raise ValueError('Unexpected review fields; automatic scores cannot be overridden here')
    if review.get('report_sha256') != base['report_sha256']:
        raise ValueError('Report hash mismatch: review belongs to a different or changed run')
    try:
        stamp = datetime.fromisoformat(review['created_at'])
        if stamp.tzinfo is None:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError('created_at must be a timezone-aware ISO timestamp') from None
    reviewer = review.get('reviewer')
    if not isinstance(reviewer, dict) or set(reviewer) != {'name', 'kind'}:
        raise ValueError('Reviewer must have name and kind')
    if not isinstance(reviewer['name'], str) or reviewer['kind'] not in ('', 'human', 'assistant'):
        raise ValueError('Reviewer kind must be human or assistant (blank only while pending)')
    rows = review.get('cases')
    if not isinstance(rows, list) or len(rows) != len(base['cases']):
        raise ValueError('Review must preserve every case, including generation errors')
    statuses, details = Counter(), []
    for row, original in zip(rows, base['cases']):
        if not isinstance(row, dict) or set(row) != set(original):
            raise ValueError('Invalid review case shape')
        for key in set(original) - {'criteria', 'notes'}:
            if digest_json(row[key]) != digest_json(original[key]):
                raise ValueError(f"Case {original['id']}: immutable field changed: {key}")
        if not isinstance(row['notes'], str):
            raise ValueError('Notes must be text')
        criteria = row['criteria']
        if not isinstance(criteria, dict) or set(criteria) != set(CRITERIA):
            raise ValueError('Review must include exactly the four reply criteria')
        results = []
        for key, value in criteria.items():
            if not isinstance(value, dict) or set(value) != {'result', 'evidence'}:
                raise ValueError('Criterion requires result and evidence')
            if not isinstance(value['result'], str) or value['result'] not in VALUES or not isinstance(value['evidence'], str):
                raise ValueError('Invalid criterion result/evidence')
            fixed_na = (not original['reviewable'] or
                        (key == 'asks_missing' and original['expected']['behavior'] != 'clarify'))
            if fixed_na:
                if value['result'] != 'not_applicable':
                    raise ValueError('Cannot score an unavailable/inapplicable reply criterion')
            else:
                if key in ('asks_missing', 'no_premature_success') and value['result'] == 'not_applicable':
                    raise ValueError('Required reply criterion cannot be skipped')
                if value['result'] != 'pending':
                    if not value['evidence'].strip():
                        raise ValueError('Reviewed criteria require evidence, including not_applicable')
                    if not reviewer['name'].strip() or not reviewer['kind']:
                        raise ValueError('Reviewed criteria require a named reviewer and kind')
            results.append(value['result'])
        status = ('not_evaluated' if not original['reviewable'] else
                  'pending' if 'pending' in results else
                  'fail' if 'fail' in results else 'pass')
        statuses[status] += 1
        details.append({'id': row['id'], 'reply_status': status, 'automatic_status': row['automatic_status']})
    return {'scope': 'Manual reply criteria only; does not change proposal scores or evaluate policy/application.',
            'reviewer': reviewer, 'reply_status': {s: statuses[s] for s in ('pending', 'pass', 'fail', 'not_evaluated')},
            'cases': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Create a pending sidecar; never overwrite an existing file')
    init.add_argument('--report', type=Path, required=True)
    init.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('check', help='Verify hashes, evidence and review completeness')
    check.add_argument('--report', type=Path, required=True)
    check.add_argument('--review', type=Path, required=True)
    check.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'init':
            data = template(args.report)
            with args.output.open('x', encoding='utf-8') as output:
                json.dump(data, output, ensure_ascii=False, indent=2)
            print(f'Created pending review: {args.output}')
            return 0
        data = strict_json(args.review.read_bytes())
        summary = validate_review(args.report, data)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if args.require_complete and (summary['reply_status']['pending'] or summary['reply_status']['not_evaluated']) else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
