"""Audit proposal splits offline; lexical similarity flags require author review."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from scripts.eval_proposals import ACTION_TO_TOOL, load_cases, strict_json

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'evals/model_proposal_development_v1.json'
HOLDOUT = ROOT / 'evals/model_proposal_holdout_v1.json'
FAMILIES = ROOT / 'evals/model_proposal_families_v1.json'
LOCK = ROOT / 'evals/model_proposal_manifest_v1.json'
HISTORY = ROOT / 'evals/model_proposal_history_v1.json'
THRESHOLD = 0.75


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape(prompt, record):
    """Mask gold payload/IDs solely for leakage screening, never for scoring."""
    values = []
    def visit(v):
        if isinstance(v, dict):
            for k, item in v.items():
                if k in ('content', 'when') and isinstance(item, str) and item:
                    values.append(item)
                visit(item)
        elif isinstance(v, list):
            for item in v:
                visit(item)
    for key in ('expected', 'expected_calls', 'expected_call_options'):
        visit(record.get(key))
    text = unicodedata.normalize('NFKC', prompt).casefold()
    for value in sorted(set(values), key=len, reverse=True):
        text = text.replace(unicodedata.normalize('NFKC', value).casefold(), ' <payload> ')
    text = re.sub(r'\d+', ' <id> ', text)
    return re.sub(r'[^\w<>]+', ' ', text).strip()


def history_records(paths):
    records = []
    def visit(v, path):
        if isinstance(v, dict):
            prompt = v.get('prompt') or v.get('user_message')
            if isinstance(prompt, str):
                records.append({'source': str(path.relative_to(ROOT)), 'id': v.get('id'),
                                'prompt': prompt, 'shape': shape(prompt, v)})
            for item in v.values():
                visit(item, path)
        elif isinstance(v, list):
            for item in v:
                visit(item, path)
    for path in paths:
        visit(strict_json(path.read_bytes()), path)
    return records


def validate_split(cases, total, per_action, controls):
    if len(cases) != total:
        raise ValueError(f'Expected {total} cases, got {len(cases)}')
    counts = Counter(c['expected']['action'] for c in cases if c['expected']['behavior'] == 'call')
    if counts != Counter({action: per_action for action in ACTION_TO_TOOL}):
        raise ValueError(f'Action balance mismatch: {dict(counts)}')
    if sum(c['expected']['behavior'] != 'call' for c in cases) != controls:
        raise ValueError('Control count mismatch')
    prompts = [c['prompt'] for c in cases]
    if len(set(prompts)) != len(prompts):
        raise ValueError('Duplicate prompt inside split')
    for c in cases:
        if not isinstance(c.get('dimensions'), list) or not c['dimensions']:
            raise ValueError('Each case needs dimensions')
    return dict(counts)


def audit(dev_path=DEV, holdout_path=HOLDOUT, families_path=FAMILIES, history_path=HISTORY):
    _, dev = load_cases(dev_path)
    _, hold = load_cases(holdout_path)
    counts = {'development': validate_split(dev, 80, 8, 24),
              'holdout': validate_split(hold, 120, 12, 36)}
    _, pilot = load_cases(ROOT / 'evals/model_proposal_pilot_v1.json')
    dev_by_id = {c['id']: c for c in dev}
    for case in pilot:
        if case['id'] not in dev_by_id or any(dev_by_id[case['id']].get(k) != v for k, v in case.items()):
            raise ValueError('Pilot cases must be preserved in development')
    if {c['id'] for c in dev} & {c['id'] for c in hold}:
        raise ValueError('IDs shared between splits')
    if {c['family'] for c in dev} & {c['family'] for c in hold}:
        raise ValueError('Family shared between splits')
    families = strict_json(Path(families_path).read_bytes())['families']
    used = {c['family'] for c in dev + hold}
    if used != set(families):
        raise ValueError('Family catalog does not match dataset')
    for split, cases in (('development', dev), ('holdout', hold)):
        for c in cases:
            f = families[c['family']]
            if f['split'] != split or not f['description'].strip():
                raise ValueError('Invalid family ownership or description')
    history = strict_json(Path(history_path).read_bytes())
    records = history['records']
    comparison = [{'source': str(DEV.relative_to(ROOT)), 'id': c['id'],
                   'prompt': c['prompt'], 'shape': shape(c['prompt'], c)} for c in dev] + records
    unique = {(r['source'], r['prompt'], r['shape']): r for r in comparison}
    comparison = list(unique.values())
    flagged, nearest, exact = [], [], []
    for case in hold:
        target = shape(case['prompt'], case)
        ranked = sorted(((SequenceMatcher(None, target, r['shape'], autojunk=False).ratio(), i)
                         for i, r in enumerate(comparison)), reverse=True)
        score, index = ranked[0]
        match = comparison[index]
        item = {'id': case['id'], 'family': case['family'], 'prompt': case['prompt'],
                'shape': target, 'similarity': round(score, 6), 'nearest': match}
        nearest.append(item)
        if score >= THRESHOLD:
            flagged.append(item)
        if any(target == r['shape'] or case['prompt'].casefold() == r['prompt'].casefold() for r in comparison):
            exact.append(case['id'])
    return {'schema_version': 1, 'method': 'NFKC/casefold; mask gold content/when and digits; SequenceMatcher. Screening only, not a semantic proof.',
            'threshold': THRESHOLD, 'counts': counts,
            'dataset_hashes': {'development': sha(dev_path), 'holdout': sha(holdout_path), 'families': sha(families_path)},
            'history_files': history['source_hashes'],
            'history_index_sha256': sha(history_path),
            'history_records': len(records), 'exact_or_masked_holdout_duplicates': exact,
            'flagged_pairs': flagged, 'nearest_pairs': nearest}



def validate_family_review(result, reviewed):
    flags = {r['id']: r for r in result['flagged_pairs']}
    decisions = reviewed['similarity_decisions']
    if len(decisions) != len(flags) or {r['id'] for r in decisions} != set(flags):
        raise ValueError('Similarity flags lack unique matching review decisions')
    for row in decisions:
        flag = flags[row['id']]
        if (row['decision'] != 'keep_distinct_framing' or
                not isinstance(row.get('reason'), str) or not row['reason'].strip()):
            raise ValueError('Similarity decision requires rationale')
        if (row['nearest_source'] != flag['nearest']['source'] or
                row['nearest_prompt'] != flag['nearest']['prompt'] or
                row['similarity'] != flag['similarity']):
            raise ValueError('Similarity decision belongs to a different pair')


def verify_lock(path=LOCK):
    manifest = strict_json(Path(path).read_bytes())
    for relative, digest in manifest['files'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Locked file changed: {relative}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-lock', action='store_true')
    args = parser.parse_args()
    try:
        manifest = verify_lock() if args.verify_lock else None
        result = audit()
        if args.output:
            with args.output.open('x', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps({k: result[k] for k in ('counts', 'history_records', 'exact_or_masked_holdout_duplicates')}, ensure_ascii=False, indent=2))
        print(f"Similarity flags: {len(result['flagged_pairs'])}")
        if manifest:
            recorded = strict_json((ROOT / manifest['audit']).read_bytes())
            if result != recorded:
                raise ValueError('Recomputed audit differs from locked audit')
            reviewed = strict_json((ROOT / manifest['family_review']).read_bytes())
            validate_family_review(result, reviewed)
            print('Locked audit matches; every flagged pair has a matching review decision.')
    except (OSError, ValueError, KeyError, TypeError) as e:
        parser.error(str(e))
    return 1 if result['exact_or_masked_holdout_duplicates'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
