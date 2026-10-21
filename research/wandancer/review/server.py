#!/usr/bin/env python3
"""Private Wan-Dancer review adapter; standard library only, loopback only.

Adapted from the repository's OmniDance review mechanisms (see README.md).
No filesystem scans, import-time media probes, external assets, or uploads.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = HERE.parents[2]
MIME = {'.mp4': 'video/mp4', '.wav': 'audio/wav', '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
STATUSES = {'pending', 'running', 'completed', 'failed', 'blocked'}
DIMS = ('sync', 'motion', 'quality')
FLAGS = {'best', 'flicker', 'collapse', 'small_motion', 'identity_drift', 'music_unrelated'}


def within(path, root):
    return path == root or root in path.parents


def identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', value):
        raise ValueError('invalid ' + label)
    return value


def secure_open(path, flags=os.O_RDONLY, mode=0o600):
    """Open a canonical absolute path without following replacement symlinks."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute canonical path required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        result = os.open(path.name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode, dir_fd=fd)
        if not stat.S_ISREG(os.fstat(result).st_mode):
            os.close(result)
            raise ValueError('regular file required')
        return result
    finally:
        os.close(fd)


def read_json(path):
    with os.fdopen(secure_open(Path(path).resolve(strict=True)), 'r', encoding='utf-8') as stream:
        return json.load(stream)


def byte_range(header, size):
    if header is None:
        return 0, size - 1, 200
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()) or size == 0:
        raise ValueError('unsatisfiable range')
    start, end = match.groups()
    if not start:
        length = int(end)
        if length == 0:
            raise ValueError('unsatisfiable range')
        return max(0, size - length), size - 1, 206
    start = int(start)
    end = min(int(end), size - 1) if end else size - 1
    if start >= size or start > end:
        raise ValueError('unsatisfiable range')
    return start, end, 206


def numeric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid ' + label)
    return value


class ReviewState:
    def __init__(self, manifest, allowed_roots, reviews, ffprobe=None, run_manifests=()):
        if not allowed_roots:
            raise ValueError('at least one --allowed-root is required')
        self.roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
        if any(not root.is_dir() or root == Path('/') for root in self.roots):
            raise ValueError('allowed roots must be explicit directories, not /')
        self.manifest_path = Path(manifest).resolve(strict=True)
        self.run_manifests = tuple(Path(p).resolve(strict=True) for p in run_manifests)
        self.reviews = Path(reviews).absolute()
        if self.reviews.is_symlink():
            raise ValueError('review output cannot be a symlink')
        self.reviews = self.reviews.parent.resolve(strict=True) / self.reviews.name
        if within(self.reviews, PUBLIC_ROOT):
            raise ValueError('review output must be outside the public checkout')
        self.ffprobe = ffprobe
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.cache = {}
        self.reload()

    def reload(self):
        data = read_json(self.manifest_path)
        if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('cases'), list):
            raise ValueError('manifest requires schema_version=1 and cases list')
        cases = json.loads(json.dumps(data['cases']))
        by_id = {}
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError('case must be an object')
            cid = identifier(case.get('case_id'), 'case_id')
            if cid in by_id:
                raise ValueError('duplicate case_id')
            case.setdefault('runs', [])
            if not isinstance(case['runs'], list):
                raise ValueError('runs must be a list')
            by_id[cid] = case
        for path in self.run_manifests:
            extra = read_json(path)
            if not isinstance(extra, dict) or extra.get('schema_version') != 1 or not isinstance(extra.get('runs'), list):
                raise ValueError('run manifest requires schema_version=1 and runs list')
            for run in extra['runs']:
                if not isinstance(run, dict) or run.get('case_id') not in by_id:
                    raise ValueError('run references unknown case_id')
                by_id[run['case_id']]['runs'].append(run)
        media, targets = {}, {}
        for case in cases:
            for field in ('reference', 'gt', 'original_audio', 'swap_audio'):
                self._register(case.get(field), media)
            seen = set()
            for run in case['runs']:
                if not isinstance(run, dict):
                    raise ValueError('run must be an object')
                rid = identifier(run.get('run_id'), 'run_id')
                if rid in seen:
                    raise ValueError('duplicate run_id within case')
                seen.add(rid)
                for field in ('model', 'version', 'condition'):
                    if not isinstance(run.get(field), str) or not run[field] or len(run[field]) > 240:
                        raise ValueError('run requires ' + field)
                if type(run.get('seed')) is not int or run['seed'] < 0:
                    raise ValueError('seed must be a nonnegative integer')
                if run.get('status') not in STATUSES:
                    raise ValueError('invalid run status')
                for stage in ('global', 'local'):
                    item = run.get(stage, {'status': 'pending'})
                    if not isinstance(item, dict) or item.get('status') not in STATUSES:
                        raise ValueError('invalid stage status')
                    if item['status'] == 'completed' and not item.get('path'):
                        raise ValueError('completed stage requires media path')
                    self._register(item.get('path'), media)
                    for key in ('crop', 'working_canvas'):
                        if key in item:
                            obj = item[key]
                            if not isinstance(obj, dict) or set(obj) != {'width', 'height'} or any(type(v) is not int or v <= 0 for v in obj.values()):
                                raise ValueError(key + ' requires positive integer width and height')
                    if 'trim' in item:
                        trim = item['trim']
                        if not isinstance(trim, dict) or set(trim) != {'head_seconds', 'tail_seconds'}:
                            raise ValueError('trim requires head_seconds and tail_seconds')
                        for val in trim.values():
                            numeric(val, 'trim')
                    expected = item.get('expected', {})
                    if not isinstance(expected, dict) or set(expected) - {'duration_seconds', 'duration_tolerance_seconds'}:
                        raise ValueError('invalid expected contract')
                    for val in expected.values():
                        numeric(val, 'expected duration')
                    key = json.dumps([case['case_id'], rid, stage], separators=(',', ':'))
                    targets[key] = (case, run, stage, item)
        with self.lock:
            self.cases, self.media, self.targets = cases, media, targets

    def _register(self, value, media):
        if value is None:
            return None
        if not isinstance(value, str) or not value or not Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('media requires absolute path without traversal')
        path = Path(value)
        real = path.resolve()
        if not any(within(real, root) for root in self.roots) or real.suffix.lower() not in MIME:
            raise ValueError('media outside allowed roots or unsupported type')
        mid = hashlib.sha256(value.encode()).hexdigest()[:32]
        media[mid] = path
        return mid

    def open_media(self, mid):
        with self.lock:
            path = self.media.get(mid)
        if path is None:
            raise FileNotFoundError('unknown media')
        real = path.resolve(strict=True)
        if not any(within(real, root) for root in self.roots):
            raise FileNotFoundError('media no longer allowed')
        return secure_open(real)

    def probe(self, mid):
        try:
            fd = self.open_media(mid)
        except (OSError, ValueError):
            return {'status': 'blocked', 'reason': 'missing_or_forbidden_media'}
        try:
            st = os.fstat(fd)
            key = (mid, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            if key in self.cache:
                return self.cache[key]
            executable = shutil.which(self.ffprobe or 'ffprobe')
            if executable is None:
                return {'status': 'blocked', 'reason': 'ffprobe_unavailable', 'action': 'Specify --ffprobe with an independent environment executable.'}
            try:
                result = subprocess.run([executable, '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', '/proc/self/fd/' + str(fd)],
                                        pass_fds=(fd,), capture_output=True, text=True, timeout=30, check=False)
                if result.returncode:
                    return {'status': 'blocked', 'reason': 'ffprobe_failed'}
                info = json.loads(result.stdout)
                videos = [s for s in info['streams'] if s.get('codec_type') == 'video']
                audios = [s for s in info['streams'] if s.get('codec_type') == 'audio']
                video = videos[0] if videos else {}
                rate = video.get('avg_frame_rate', '0/1').split('/')
                fps = float(rate[0]) / float(rate[1]) if float(rate[1]) else 0
                count = video.get('nb_read_frames', video.get('nb_frames'))
                duration = float(info.get('format', {}).get('duration', video.get('duration', '0')))
                measured = {'status': 'measured', 'width': video.get('width'), 'height': video.get('height'),
                            'fps': fps, 'frame_count': int(count) if count not in (None, 'N/A') else None,
                            'duration_seconds': duration, 'has_audio': bool(audios), 'has_video': bool(videos)}
                if not math.isfinite(fps) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError('invalid probe numbers')
            except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, IndexError):
                return {'status': 'blocked', 'reason': 'ffprobe_failed_or_incomplete'}
            self.cache[key] = measured
            return measured
        finally:
            os.close(fd)

    def media_info(self, value):
        if value is None:
            return None
        mid = hashlib.sha256(value.encode()).hexdigest()[:32]
        return {'media_id': mid, 'probe': self.probe(mid)}

    def validate_stage(self, case, run, stage, item):
        if item['status'] != 'completed':
            return {'status': 'blocked', 'reason': 'stage_' + item['status']}
        metadata = self.media_info(item.get('path'))
        measured = metadata['probe']
        if measured['status'] != 'measured':
            return measured
        problems = []
        if not measured['has_video']:
            problems.append('video_stream_required')
        if abs(measured['fps'] - (8 if stage == 'global' else 30)) > 0.01:
            problems.append('fps_mismatch')
        if stage == 'global':
            if measured['frame_count'] is None:
                return {'status': 'blocked', 'reason': 'global_frame_count_unverified'}
            if measured['frame_count'] != 38:
                problems.append('global_requires_38_frames')
            if measured['has_audio']:
                problems.append('global_must_not_contain_audio')
            if abs(measured['duration_seconds'] - 38 / 8) > 1 / 8:
                problems.append('global_container_duration_mismatch')
        else:
            if not measured['has_audio']:
                problems.append('local_audio_required')
            crop = item.get('crop')
            trim = item.get('trim')
            if crop is None or trim is None:
                return {'status': 'blocked', 'reason': 'local_crop_and_trim_contract_required'}
            if (measured['width'], measured['height']) != (crop['width'], crop['height']):
                problems.append('local_crop_size_mismatch')
            audio = case.get('swap_audio' if run['condition'] in ('aswap', 'swap') else 'original_audio')
            if audio is None:
                return {'status': 'blocked', 'reason': 'condition_audio_required'}
            audio_probe = self.media_info(audio)['probe']
            if audio_probe['status'] != 'measured':
                return {'status': 'blocked', 'reason': 'condition_audio_probe_unavailable'}
            duration = audio_probe['duration_seconds'] - trim['head_seconds'] - trim['tail_seconds']
            expected = item.get('expected', {})
            if 'duration_seconds' in expected and abs(expected['duration_seconds'] - duration) > 1 / 30:
                problems.append('declared_duration_trim_mismatch')
            if abs(measured['duration_seconds'] - duration) > expected.get('duration_tolerance_seconds', 1 / 30 + 0.005):
                problems.append('local_trimmed_duration_mismatch')
        return {'status': 'failed' if problems else 'passed', 'problems': problems}

    def public_manifest(self):
        with self.lock:
            cases = []
            for case in self.cases:
                out = {'case_id': case['case_id'], 'runs': []}
                for field in ('reference', 'gt', 'original_audio', 'swap_audio'):
                    out[field] = self.media_info(case.get(field))
                for run in case['runs']:
                    record = {k: run[k] for k in ('run_id', 'model', 'version', 'seed', 'condition', 'status')}
                    for stage in ('global', 'local'):
                        item = run.get(stage, {'status': 'pending'})
                        record[stage] = {k: item[k] for k in ('status', 'crop', 'trim', 'working_canvas', 'expected') if k in item}
                        record[stage]['media'] = self.media_info(item.get('path'))
                        record[stage]['validation'] = self.validate_stage(case, run, stage, item)
                    out['runs'].append(record)
                cases.append(out)
            return {'schema_version': 1, 'cases': cases, 'review_token': self.token,
                    'notice': 'Local CPU checks are not deployment or human review acceptance.'}

    def save_review(self, data):
        if not isinstance(data, dict) or set(data) - {'reviewer', 'case_id', 'run_id', 'stage', *DIMS, 'flags', 'note'}:
            raise ValueError('invalid review fields')
        reviewer = identifier(data.get('reviewer'), 'reviewer alias')
        cid = identifier(data.get('case_id'), 'case_id')
        rid = identifier(data.get('run_id'), 'run_id')
        stage = data.get('stage')
        if stage not in ('global', 'local'):
            raise ValueError('invalid stage')
        key = json.dumps([cid, rid, stage], separators=(',', ':'))
        with self.lock:
            if key not in self.targets:
                raise ValueError('unknown review target')
            case, run, _, item = self.targets[key]
            if item['status'] != 'completed' or not item.get('path'):
                raise ValueError('cannot review an incomplete stage')
            try:
                fd = self.open_media(hashlib.sha256(item['path'].encode()).hexdigest()[:32])
                os.close(fd)
            except (OSError, ValueError):
                raise ValueError('review target media unavailable') from None
            for dim in DIMS:
                val = data.get(dim)
                if val is not None and (type(val) is not int or not 1 <= val <= 5):
                    raise ValueError('scores must be integer 1..5 or null')
            flags = data.get('flags', [])
            if not isinstance(flags, list) or any(not isinstance(f, str) or f not in FLAGS for f in flags) or len(flags) != len(set(flags)):
                raise ValueError('invalid flags')
            note = data.get('note', '')
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError('invalid note')
            if not any(data.get(d) is not None for d in DIMS) and not flags and not note:
                raise ValueError('empty review')
            record = {**data, 'reviewer': reviewer, 'flags': flags, 'note': note, 'key': key,
                      'model': run['model'], 'version': run['version'], 'seed': run['seed'],
                      'condition': run['condition'], 'timestamp_ns': time.time_ns()}
            record['validation'] = self.validate_stage(case, run, stage, item)
            line = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
            fd = secure_open(self.reviews, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                with os.fdopen(fd, 'ab', closefd=False) as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
            return record

    def load_reviews(self):
        with self.lock:
            if not self.reviews.exists():
                return []
            fd = secure_open(self.reviews)
            with os.fdopen(fd, 'r', encoding='utf-8') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
                records = {}
                for line in stream:
                    try:
                        item = json.loads(line)
                        records[(item['reviewer'], item['key'])] = item
                    except (ValueError, KeyError, TypeError):
                        continue
                return list(records.values())


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, state):
        self.state = state
        super().__init__(('127.0.0.1', port), Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def trusted_host(self):
        host = self.headers.get('Host', '')
        port = self.server.server_address[1]
        return host in {f'127.0.0.1:{port}', f'localhost:{port}'}

    def send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        for key, value in {'Content-Type': ctype, 'Content-Length': str(len(body)),
                           'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                           'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'self'", **(extra or {})}.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.trusted_host():
            return self.send(403, {'error': 'loopback Host required'})
        url = urlsplit(self.path)
        try:
            if url.path in ('/', '/index.html', '/app.js', '/style.css'):
                name = 'index.html' if url.path == '/' else url.path[1:]
                ctype = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript; charset=utf-8', 'style.css': 'text/css; charset=utf-8'}[name]
                return self.send(200, (HERE / name).read_bytes(), ctype)
            if url.path == '/api/manifest':
                return self.send(200, self.server.state.public_manifest())
            if url.path == '/api/reviews':
                return self.send(200, self.server.state.load_reviews())
            if url.path == '/media':
                query = parse_qs(url.query)
                if set(query) != {'id'} or len(query['id']) != 1:
                    return self.send(404, {'error': 'unknown media'})
                return self.serve_media(query['id'][0])
            return self.send(404, {'error': 'not found'})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError):
            self.send(503, {'status': 'blocked', 'reason': 'private_file_unavailable'})

    def serve_media(self, mid):
        try:
            fd = self.server.state.open_media(mid)
        except (OSError, ValueError):
            return self.send(404, {'error': 'unknown or unavailable media'})
        with os.fdopen(fd, 'rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            try:
                start, end, code = byte_range(self.headers.get('Range'), size)
            except ValueError:
                return self.send(416, b'', extra={'Content-Range': f'bytes */{size}', 'Accept-Ranges': 'bytes'})
            self.send_response(code)
            self.send_header('Content-Type', MIME[self.server.state.media[mid].suffix.lower()])
            self.send_header('Content-Length', str(max(0, end - start + 1)))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if code == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            stream.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = stream.read(min(left, 1024 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_POST(self):
        self.close_connection = True
        if not self.trusted_host() or not secrets.compare_digest(self.headers.get('X-Review-Token', ''), self.server.state.token):
            return self.send(403, {'error': 'local review token required'})
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_address[1]}', f'http://localhost:{self.server.server_address[1]}'}:
            return self.send(403, {'error': 'cross-origin write forbidden'})
        url = urlsplit(self.path)
        if url.path not in ('/api/review', '/api/reload'):
            return self.send(404, {'error': 'not found'})
        if self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding'):
            return self.send(415, {'error': 'bounded JSON body required'})
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            return self.send(411, {'error': 'Content-Length required'})
        if not 0 < length <= 16384:
            return self.send(413, {'error': 'invalid body size'})
        self.connection.settimeout(10)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('incomplete body')
            data = json.loads(body)
            if url.path == '/api/reload':
                if data != {}:
                    raise ValueError('reload body must be empty object')
                self.server.state.reload()
                return self.send(200, {'ok': True})
            record = self.server.state.save_review(data)
            return self.send(200, {'ok': True, 'review': record})
        except (ValueError, TypeError, UnicodeError):
            self.send(400, {'error': 'invalid JSON, review fields, or manifest contract'})
        except (OSError, TimeoutError):
            self.send(503, {'status': 'blocked', 'reason': 'private_output_unavailable'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='External private schema_version=1 case manifest')
    parser.add_argument('--run-manifest', action='append', default=[], help='External private run manifest; repeatable')
    parser.add_argument('--allowed-root', action='append', required=True, help='Explicit media directory; repeatable')
    parser.add_argument('--reviews', required=True, help='Private output JSONL; parent directory must already exist')
    parser.add_argument('--ffprobe', help='ffprobe executable from a separate environment')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args(argv)
    try:
        state = ReviewState(args.manifest, args.allowed_root, args.reviews, args.ffprobe, args.run_manifest)
    except (OSError, ValueError, TypeError):
        parser.error('blocked: invalid/unavailable private manifest, allowed roots, or review output; consult README contract')
    with ReviewServer(args.port, state) as server:
        print(f'Local private review listening on 127.0.0.1:{args.port}; no external upload.', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
