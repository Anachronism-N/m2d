"""Private rating coverage and gated snapshots; never infer human acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from server import DIMS, identifier, read_json, secure_open
import os


def coverage(manifest, records, reviewers):
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('cases'), list):
        raise ValueError('Versioned case manifest required')
    if not reviewers or len(set(reviewers)) != len(reviewers):
        raise ValueError('Explicit unique reviewer aliases required')
    for reviewer in reviewers:
        identifier(reviewer, 'reviewer')
    targets = set()
    for case in manifest['cases']:
        cid = identifier(case['case_id'], 'case_id')
        for run in case['runs']:
            rid = identifier(run['run_id'], 'run_id')
            local = run.get('local', {})
            if run['status'] != 'completed' or local.get('status') != 'completed':
                raise ValueError('All planned Local outputs must be completed before review')
            key = (cid, rid, 'local')
            if key in targets:
                raise ValueError('Duplicate review target')
            targets.add(key)
    if not targets:
        raise ValueError('Nonempty planned target set required')
    latest = {}
    for row in records:
        key = (row.get('case_id'), row.get('run_id'), row.get('stage'))
        if key not in targets:
            raise ValueError('Review references an unknown or non-Local target')
        identifier(row.get('reviewer'), 'reviewer')
        if row['reviewer'] in reviewers:
            latest[(row['reviewer'], key)] = row
    rows = []
    selected = []
    for reviewer in reviewers:
        missing = []
        for key in sorted(targets):
            row = latest.get((reviewer, key))
            problems = []
            if row is None:
                problems.append('not_reviewed')
            else:
                for dim in DIMS:
                    value = row.get(dim)
                    if type(value) is not int or not 1 <= value <= 5:
                        problems.append('missing_or_invalid_' + dim)
                if not isinstance(row.get('note'), str) or not row['note'].strip():
                    problems.append('observable_fact_note_required')
                if row.get('validation', {}).get('status') != 'passed':
                    problems.append('saved_media_validation_not_passed')
            if problems:
                missing.append({'case_id': key[0], 'run_id': key[1], 'reasons': problems})
            else:
                selected.append(row)
        rows.append({'reviewer': reviewer, 'expected': len(targets),
                     'complete': len(targets) - len(missing), 'missing': missing})
    return {'schema_version': 1, 'expected_variants': len(targets),
            'required_reviewers': len(reviewers), 'coverage_complete': all(not x['missing'] for x in rows),
            'reviewers': rows, 'selected_reviews': selected,
            'independent_reviewer_identity_verified': False,
            'quality_or_music_success_approved': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--reviews', required=True, type=Path)
    parser.add_argument('--reviewer', required=True, action='append')
    parser.add_argument('--freeze', type=Path, help='New private snapshot, only after full rating coverage')
    args = parser.parse_args(argv)
    manifest = read_json(args.manifest)
    raw = b''
    if args.reviews.is_symlink():
        parser.error('Ratings cannot be a symlink')
    if args.reviews.exists():
        with os.fdopen(secure_open(args.reviews.resolve()), 'rb') as stream:
            raw = stream.read()
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        report = coverage(manifest, records, args.reviewer)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.freeze:
        if not report['coverage_complete']:
            parser.error('Incomplete ratings: refuse freezing or unblinding')
        destination = args.freeze.absolute()
        destination = destination.parent.resolve(strict=True) / destination.name
        if destination.is_relative_to(Path(__file__).resolve().parents[3]):
            parser.error('Snapshot must be outside publication checkout')
        snapshot = {**report, 'frozen_time_ns': time.time_ns(),
                    'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    'ratings_source_sha256': hashlib.sha256(raw).hexdigest(),
                    'warning': 'Frozen ratings are not proof of independent human identity, valid reward or quality improvement'}
        with os.fdopen(secure_open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write(chr(10))
        report['snapshot_written'] = True
    report.pop('selected_reviews')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
