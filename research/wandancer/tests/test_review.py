import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import unittest
import urllib.error
import urllib.request

p = Path(__file__).resolve().parents[1] / 'review' / 'server.py'
spec = importlib.util.spec_from_file_location('review_server', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

analysis_path = p.parent / 'analyze_reviews.py'
analysis_spec = importlib.util.spec_from_file_location('review_analysis', analysis_path)
analysis = importlib.util.module_from_spec(analysis_spec)
sys.path.insert(0, str(p.parent))
try:
    analysis_spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / 'media'
        self.media.mkdir()
        self.video = self.media / 'generated.mp4'
        self.video.write_bytes(b'0123456789')
        self.manifest = self.root / 'manifest.json'
        self.data = {'schema_version': 1, 'cases': [{
            'case_id': 'fixture', 'reference': None, 'gt': None,
            'original_audio': None, 'swap_audio': None, 'runs': [{
                'run_id': 'fixture-run', 'model': 'test', 'version': 'fixture',
                'condition': 'joint', 'seed': 0, 'status': 'completed',
                'global': {'status': 'completed', 'path': str(self.video)},
                'local': {'status': 'pending'}}]}]}
        self.manifest.write_text(json.dumps(self.data))
        self.state = module.ReviewState(self.manifest, [self.media],
                                       self.root / 'reviews.jsonl', ffprobe='/missing/ffprobe')

    def test_exact_proxy_origin_only(self):
        state = module.ReviewState(self.manifest, [self.media], self.root / 'proxy_reviews.jsonl',
                                   trusted_origin='https://review.example.invalid')
        self.assertEqual(state.trusted_origin, 'https://review.example.invalid')
        self.assertEqual(state.trusted_proxy_host, 'review.example.invalid')
        server = module.ReviewServer(0, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = 'http://127.0.0.1:' + str(server.server_address[1]) + '/api/manifest'
        request = urllib.request.Request(url, headers={'Host': 'review.example.invalid'})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)
        request = urllib.request.Request(url, headers={'Host': 'other.example.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)
        for origin in ('http://review.example.invalid', 'https://review.example.invalid/path',
                       'https://user:secret@review.example.invalid'):
            with self.assertRaises(ValueError):
                module.ReviewState(self.manifest, [self.media], self.root / 'invalid.jsonl',
                                   trusted_origin=origin)

    def test_range_bounds(self):
        self.assertEqual(module.byte_range('bytes=2-4', 10), (2, 4, 206))
        self.assertEqual(module.byte_range('bytes=-3', 10), (7, 9, 206))
        self.assertEqual(module.byte_range('bytes=8-', 10), (8, 9, 206))
        for value in ('bytes=10-', 'bytes=3-2', 'bytes=0-1,3-4', 'bytes=-0'):
            with self.assertRaises(ValueError):
                module.byte_range(value, 10)

    def test_unknown_path_and_symlink_escape(self):
        with self.assertRaises(FileNotFoundError):
            self.state.open_media('unknown')
        outside = self.root / 'outside.mp4'
        outside.write_bytes(b'private')
        self.video.unlink()
        self.video.symlink_to(outside)
        mid = next(iter(self.state.media))
        with self.assertRaises(FileNotFoundError):
            self.state.open_media(mid)

    def test_registration_rejects_escape(self):
        self.data['cases'][0]['runs'][0]['global']['path'] = str(self.root / 'outside.mp4')
        self.manifest.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            self.state.reload()

    def test_missing_probe_is_blocked_not_passed(self):
        result = self.state.public_manifest()['cases'][0]['runs'][0]['global']['validation']
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'ffprobe_unavailable')

    def test_invalid_score_and_pending_stage_rejected(self):
        data = {'reviewer': 'reviewer', 'case_id': 'fixture', 'run_id': 'fixture-run',
                'stage': 'global', 'quality': True}
        with self.assertRaises(ValueError):
            self.state.save_review(data)
        data.update(stage='local', quality=4)
        with self.assertRaises(ValueError):
            self.state.save_review(data)

    def test_private_review_saved_with_blocked_validation(self):
        data = {'reviewer': 'alias', 'case_id': 'fixture', 'run_id': 'fixture-run',
                'stage': 'global', 'quality': 4, 'note': 'synthetic fixture'}
        result = self.state.save_review(data)
        self.assertEqual(result['validation']['status'], 'blocked')
        self.assertEqual(len(self.state.load_reviews()), 1)
        self.assertEqual((self.root / 'reviews.jsonl').stat().st_mode & 0o777, 0o600)

    def test_rating_coverage_empty_is_incomplete(self):
        self.data['cases'][0]['runs'][0]['local'] = {'status': 'completed', 'path': str(self.video)}
        report = analysis.coverage(self.data, [], ['rater_A'])
        self.assertFalse(report['coverage_complete'])
        self.assertEqual(report['reviewers'][0]['complete'], 0)
        self.assertEqual(report['reviewers'][0]['missing'][0]['reasons'], ['not_reviewed'])

    def test_latest_partial_rating_does_not_reuse_old_score(self):
        self.data['cases'][0]['runs'][0]['local'] = {'status': 'completed', 'path': str(self.video)}
        complete = {'reviewer': 'rater_A', 'case_id': 'fixture', 'run_id': 'fixture-run',
                    'stage': 'local', 'sync': 3, 'motion': 4, 'quality': 4,
                    'note': 'Synthetic visible fact', 'validation': {'status': 'passed'}}
        partial = {**complete, 'quality': None}
        report = analysis.coverage(self.data, [complete, partial], ['rater_A'])
        self.assertFalse(report['coverage_complete'])
        self.assertIn('missing_or_invalid_quality', report['reviewers'][0]['missing'][0]['reasons'])

    def test_coverage_requires_every_reviewer_and_fact_notes(self):
        self.data['cases'][0]['runs'][0]['local'] = {'status': 'completed', 'path': str(self.video)}
        row = {'reviewer': 'rater_A', 'case_id': 'fixture', 'run_id': 'fixture-run',
               'stage': 'local', 'sync': 3, 'motion': 4, 'quality': 4,
               'note': 'Synthetic observed fact', 'validation': {'status': 'passed'}}
        report = analysis.coverage(self.data, [row], ['rater_A', 'rater_B'])
        self.assertFalse(report['coverage_complete'])
        report = analysis.coverage(self.data, [row, {**row, 'reviewer': 'rater_B'}], ['rater_A', 'rater_B'])
        self.assertTrue(report['coverage_complete'])
        self.assertFalse(report['quality_or_music_success_approved'])
        self.assertFalse(report['independent_reviewer_identity_verified'])
        report = analysis.coverage(self.data, [{**row, 'note': ' '}], ['rater_A'])
        self.assertFalse(report['coverage_complete'])

    def test_unknown_rating_targets_and_boolean_scores_rejected(self):
        self.data['cases'][0]['runs'][0]['local'] = {'status': 'completed', 'path': str(self.video)}
        with self.assertRaises(ValueError):
            analysis.coverage(self.data, [{'case_id': 'unknown', 'run_id': 'x', 'stage': 'local'}], ['rater_A'])
        row = {'reviewer': 'rater_A', 'case_id': 'fixture', 'run_id': 'fixture-run',
               'stage': 'local', 'sync': True, 'motion': 4, 'quality': 4,
               'note': 'Synthetic fact', 'validation': {'status': 'passed'}}
        report = analysis.coverage(self.data, [row], ['rater_A'])
        self.assertFalse(report['coverage_complete'])
        report = analysis.coverage(self.data, [{**row, 'sync': 3, 'validation': {'status': 'blocked'}}], ['rater_A'])
        self.assertFalse(report['coverage_complete'])

    def test_freeze_requires_complete_ratings_and_preserves_existing(self):
        self.data['cases'][0]['runs'][0]['local'] = {'status': 'completed', 'path': str(self.video)}
        self.manifest.write_text(json.dumps(self.data))
        ratings = self.root / 'independent.jsonl'
        snapshot = self.root / 'snapshot.json'
        command = [sys.executable, str(analysis_path), '--manifest', str(self.manifest),
                   '--reviews', str(ratings), '--reviewer', 'rater_A', '--freeze', str(snapshot)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(snapshot.exists())
        row = {'reviewer': 'rater_A', 'case_id': 'fixture', 'run_id': 'fixture-run',
               'stage': 'local', 'sync': 3, 'motion': 4, 'quality': 4,
               'note': 'Synthetic observed fact', 'validation': {'status': 'passed'}}
        ratings.write_text(json.dumps(row) + chr(10))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = snapshot.read_bytes()
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
        self.assertFalse(json.loads(before)['quality_or_music_success_approved'])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot.read_bytes(), before)

    def test_http_range_and_untrusted_host(self):
        server = module.ReviewServer(0, self.state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = 'http://127.0.0.1:' + str(server.server_address[1])
        mid = next(iter(self.state.media))
        request = urllib.request.Request(base + '/media?id=' + mid,
                                         headers={'Range': 'bytes=2-4'})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'234')
        request = urllib.request.Request(base + '/api/manifest',
                                         headers={'Host': 'untrusted.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
