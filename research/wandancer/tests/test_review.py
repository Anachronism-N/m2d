import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

p = Path(__file__).resolve().parents[1] / 'review' / 'server.py'
spec = importlib.util.spec_from_file_location('review_server', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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
