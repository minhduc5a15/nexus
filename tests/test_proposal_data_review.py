import contextlib
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from scripts import audit_proposal_data as data
from scripts import eval_proposals as evaluator
from scripts import review_proposals as review
from nexus.core.deadlines import parse_deadline


def response(reply):
    return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': reply}}]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'report.json'
        _, cases = evaluator.load_cases(evaluator.DATASET_PATH)
        case = deepcopy(next(c for c in cases if c['id'] == 'missing_deadline_time'))
        case['known_fields'] = {'id': 2}
        result = evaluator.evaluate_case(case, lambda _: response('Đã đặt hạn!'))
        self.report = {'scoring_version': 'proposal-v1', 'cases': [result]}
        self.write_report()

    def write_report(self):
        self.path.write_text(json.dumps(self.report, ensure_ascii=False))

    def complete(self, value=None):
        value = deepcopy(value) if value else review.template(self.path)
        value['reviewer'] = {'name': 'Test reviewer', 'kind': 'human'}
        for row in value['cases']:
            for name, criterion in row['criteria'].items():
                if criterion['result'] == 'pending':
                    criterion.update(result='pass', evidence='Đã đọc response và đối chiếu yêu cầu.')
        return value

    def test_template_is_pending_and_does_not_change_source(self):
        before = self.path.read_bytes()
        with patch('sqlite3.connect', side_effect=AssertionError('No SQLite')), patch('scripts.eval_proposals.send_chat', side_effect=AssertionError('No model')):
            value = review.template(self.path)
            summary = review.validate_review(self.path, value)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(summary['reply_status']['pending'], 1)
        self.assertEqual(value['cases'][0]['known_fields'], {'id': 2})
        self.assertEqual(value['cases'][0]['expected']['missing_slots'], ['when'])

    def test_reply_failure_does_not_override_automatic_proposal_pass(self):
        value = self.complete()
        value['cases'][0]['criteria']['no_premature_success'] = {'result': 'fail', 'evidence': 'Nói Đã đặt hạn! trước khi có tool result.'}
        summary = review.validate_review(self.path, value)
        self.assertEqual(summary['reply_status']['fail'], 1)
        self.assertEqual(summary['cases'][0]['automatic_status'], 'pass')
        self.assertEqual(json.loads(self.path.read_text()), self.report)

    def test_rejects_different_run_even_with_same_case_ids(self):
        value = self.complete()
        self.report['cases'][0]['raw_response'] = response('Bạn muốn đặt hạn lúc nào?')
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            review.validate_review(self.path, value)

    def test_byte_change_to_report_invalidates_review(self):
        value = review.template(self.path)
        self.path.write_text(self.path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            review.validate_review(self.path, value)

    def test_context_response_and_score_fields_are_immutable(self):
        for key, replacement in [('reply', 'Một câu khác'), ('known_fields', {}),
                                 ('response_sha256', 'bad'), ('context_sha256', 'bad'),
                                 ('automatic_status', 'fail'), ('id', 'different')]:
            with self.subTest(key=key):
                value = self.complete()
                value['cases'][0][key] = replacement
                with self.assertRaisesRegex(ValueError, 'immutable field'):
                    review.validate_review(self.path, value)

    def test_missing_rows_duplicate_rows_and_reordering_are_rejected(self):
        other = deepcopy(self.report['cases'][0]); other['id'] = 'other'
        self.report['cases'].append(other); self.write_report()
        for mutation in ('drop', 'duplicate', 'reverse'):
            with self.subTest(mutation=mutation):
                value = review.template(self.path)
                if mutation == 'drop': value['cases'].pop()
                if mutation == 'duplicate': value['cases'][1] = deepcopy(value['cases'][0])
                if mutation == 'reverse': value['cases'].reverse()
                with self.assertRaises(ValueError): review.validate_review(self.path, value)

    def test_missing_evidence_or_reviewer_is_rejected(self):
        for mutation in ('evidence', 'name', 'kind'):
            with self.subTest(mutation=mutation):
                value = self.complete()
                if mutation == 'evidence': value['cases'][0]['criteria']['asks_missing']['evidence'] = ' '
                else: value['reviewer'][mutation] = ''
                with self.assertRaises(ValueError): review.validate_review(self.path, value)

    def test_required_criteria_cannot_be_skipped(self):
        for criterion in ('asks_missing', 'no_premature_success'):
            value = self.complete()
            value['cases'][0]['criteria'][criterion]['result'] = 'not_applicable'
            with self.assertRaisesRegex(ValueError, 'cannot be skipped'):
                review.validate_review(self.path, value)

    def test_examples_can_be_inapplicable_only_with_explanation(self):
        value = self.complete()
        value['cases'][0]['criteria']['examples_supported'] = {'result': 'not_applicable', 'evidence': 'Reply không đưa ví dụ.'}
        self.assertEqual(review.validate_review(self.path, value)['reply_status']['pass'], 1)

    def test_partial_review_is_not_reported_as_pass(self):
        value = self.complete()
        value['cases'][0]['criteria']['examples_supported'] = {'result': 'pending', 'evidence': ''}
        self.assertEqual(review.validate_review(self.path, value)['reply_status']['pending'], 1)

    def test_generation_error_is_not_reviewed_or_counted_as_pass(self):
        result = self.report['cases'][0]
        result.update(automatic_status='error', raw_response=None)
        self.write_report()
        value = review.template(self.path)
        summary = review.validate_review(self.path, value)
        self.assertEqual(summary['reply_status']['not_evaluated'], 1)
        value['cases'][0]['criteria']['no_premature_success']['result'] = 'pass'
        with self.assertRaises(ValueError): review.validate_review(self.path, value)

    def test_historical_report_without_known_fields_is_supported(self):
        del self.report['cases'][0]['known_fields']; self.write_report()
        value = review.template(self.path)
        self.assertEqual(value['cases'][0]['known_fields'], {})
        self.assertEqual(review.validate_review(self.path, value)['reply_status']['pending'], 1)

    def test_invalid_reports_and_criterion_types_are_rejected(self):
        self.report['cases'].append(deepcopy(self.report['cases'][0])); self.write_report()
        with self.assertRaisesRegex(ValueError, 'unique'): review.template(self.path)
        self.report['cases'].pop(); self.write_report()
        value = self.complete()
        value['cases'][0]['criteria']['asks_missing']['result'] = True
        with self.assertRaises(ValueError): review.validate_review(self.path, value)

    def test_cli_exclusive_output_and_require_complete(self):
        output = Path(self.tmp.name) / 'review.json'
        args = ['review', 'init', '--report', str(self.path), '--output', str(output)]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(review.main(), 0)
        before = output.read_bytes()
        with patch('sys.argv', args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            review.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(output.read_bytes(), before)
        check = ['review', 'check', '--report', str(self.path), '--review', str(output), '--require-complete']
        with patch('sys.argv', check), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(review.main(), 1)
        output.write_text(json.dumps(self.complete(), ensure_ascii=False))
        with patch('sys.argv', check), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(review.main(), 0)

    def test_known_fields_metadata_does_not_enter_model_request(self):
        _, cases = evaluator.load_cases(evaluator.DATASET_PATH)
        case = deepcopy(cases[0]); case['known_fields'] = {'content': 'sentinel_label_only'}
        result = evaluator.evaluate_case(case, lambda _: response('Chờ xử lý.'))
        self.assertEqual(result['known_fields'], case['known_fields'])
        self.assertNotIn('sentinel_label_only', json.dumps(result['model_request']))


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.dev = evaluator.load_cases(data.DEV)
        _, cls.hold = evaluator.load_cases(data.HOLDOUT)

    def draft(self, cases):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'cases.json'
        path.write_text(json.dumps({'schema_version': 1, 'cases': cases}, ensure_ascii=False))
        return path

    def test_balance_preserved_pilot_and_separate_families_without_inference(self):
        with patch('sqlite3.connect', side_effect=AssertionError('No DB')), patch('scripts.eval_proposals.send_chat', side_effect=AssertionError('No inference')):
            result = data.audit()
        self.assertEqual(result['exact_or_masked_holdout_duplicates'], [])
        self.assertEqual(set(result['counts']['development'].values()), {8})
        self.assertEqual(set(result['counts']['holdout'].values()), {12})
        self.assertEqual(len(result['flagged_pairs']), 5)

    def test_gold_payloads_are_verbatim_and_deadlines_are_supported(self):
        for case in self.dev + self.hold:
            for call in case['expected']['accepted_calls']:
                args = call['arguments']
                for field in ('content', 'when'):
                    if field in args:
                        self.assertIn(args[field], case['prompt'], case['id'])
                if 'when' in args:
                    self.assertIsInstance(parse_deadline(args['when'], reference=datetime(2026, 9, 29, 10, tzinfo=ZoneInfo('Asia/Ho_Chi_Minh'))), int)

    def test_family_collision_is_rejected(self):
        cases = deepcopy(self.hold); cases[0]['family'] = self.dev[0]['family']
        with self.assertRaisesRegex(ValueError, 'Family shared'):
            data.audit(holdout_path=self.draft(cases))

    def test_id_collision_is_rejected(self):
        cases = deepcopy(self.hold); cases[0]['id'] = self.dev[0]['id']
        with self.assertRaisesRegex(ValueError, 'IDs shared'):
            data.audit(holdout_path=self.draft(cases))

    def test_masked_payload_duplicate_is_detected_across_splits(self):
        cases = deepcopy(self.hold)
        cases[0]['prompt'] = 'Lưu việc: một payload khác.'
        cases[0]['expected']['accepted_calls'][0]['arguments']['content'] = 'một payload khác.'
        result = data.audit(holdout_path=self.draft(cases))
        self.assertIn(cases[0]['id'], result['exact_or_masked_holdout_duplicates'])

    def test_pilot_label_changes_are_rejected(self):
        cases = deepcopy(self.dev)
        cases[0]['expected']['accepted_calls'][0]['arguments']['content'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Pilot cases'):
            data.audit(dev_path=self.draft(cases))

    def test_wrong_action_balance_is_rejected(self):
        cases = deepcopy(self.dev)
        cases[-1]['expected'] = {'behavior': 'call', 'action': 'list', 'accepted_calls': [{'name': 'list_tasks', 'arguments': {}}], 'missing_slots': []}
        with self.assertRaisesRegex(ValueError, 'balance'):
            data.audit(dev_path=self.draft(cases))

    def test_similarity_reviews_are_bound_to_pairs_and_need_rationale(self):
        result = json.loads((data.ROOT / 'evals/model_proposal_overlap_v1.json').read_text())
        base = json.loads((data.ROOT / 'evals/model_proposal_family_review_v1.json').read_text())
        data.validate_family_review(result, base)
        for mutation in ('missing', 'wrong_pair', 'empty_reason', 'duplicate'):
            with self.subTest(mutation=mutation):
                value = deepcopy(base)
                if mutation == 'missing': value['similarity_decisions'].pop()
                if mutation == 'wrong_pair': value['similarity_decisions'][0]['nearest_prompt'] = 'another prompt'
                if mutation == 'empty_reason': value['similarity_decisions'][0]['reason'] = ' '
                if mutation == 'duplicate': value['similarity_decisions'][1] = deepcopy(value['similarity_decisions'][0])
                with self.assertRaises(ValueError): data.validate_family_review(result, value)

    def test_hash_lock_detects_even_whitespace_change(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        locked = root / 'dataset.json'; locked.write_text('{}')
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'files': {str(locked): data.sha(locked)}}))
        data.verify_lock(manifest)
        locked.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'Locked file changed'):
            data.verify_lock(manifest)


if __name__ == '__main__':
    unittest.main()
