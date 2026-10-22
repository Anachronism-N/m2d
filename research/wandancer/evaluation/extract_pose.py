"""Read-only explicit video manifest to private NLF pose observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch
import torchvision  # Registers operators required by the NLF TorchScript model.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--gpu-memory-fraction', type=float, default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    root = args.output_root.resolve()
    publication_root = Path(__file__).resolve().parents[3]
    if root.is_relative_to(publication_root):
        parser.error('outputs must be private and outside the publication checkout')
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    if args.gpu_memory_fraction is not None:
        if args.device != 'cuda' or not 0 < args.gpu_memory_fraction <= 1:
            parser.error('gpu-memory-fraction requires CUDA and a fraction in (0,1]')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('videos'), list):
        raise ValueError('versioned explicit videos manifest required')
    records = []
    seen = set()
    for item in manifest['videos']:
        sid = item['sample_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', sid) or sid in seen:
            raise ValueError('safe unique sample_id required')
        seen.add(sid)
        video_path = Path(item['video']).resolve(strict=True)
        record = {'sample_id': sid, 'video': str(video_path), 'status': 'running',
                  'model': str(args.model), 'device': args.device,
                  'identity_tracking': False, 'person_selection': 'largest_box_each_frame',
                  'rgb_quality_verified': False, 'started_seconds': time.time()}
        records.append(record)
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('invalid decoded fps')
            frames = []
            timestamps = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            cap.release()
            if not frames or len(frames) < 3:
                raise ValueError('insufficient decoded video frames')
            times = np.asarray(timestamps, dtype=np.float64)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('decoded timestamps missing or not strictly increasing')
            count = len(frames)
            joints = np.full((count, 24, 3), np.nan, np.float32)
            pose = np.full((count, 72), np.nan, np.float32)
            boxes = np.full((count, 5), np.nan, np.float32)
            uncertainty = np.full((count, 24), np.nan, np.float32)
            valid = np.zeros(count, dtype=bool)
            with torch.inference_mode():
                for start in range(0, count, args.batch_size):
                    batch = torch.from_numpy(np.stack(frames[start:start + args.batch_size])).permute(0, 3, 1, 2).to(args.device)
                    result = model.detect_smpl_batched(batch)
                    if 'boxes' not in result:
                        if any(x.numel() for x in result['joints3d']):
                            raise ValueError('populated joints returned without boxes')
                        continue
                    for offset, candidates in enumerate(result['boxes']):
                        if len(candidates) == 0:
                            continue
                        selected = int(torch.argmax(candidates[:, 2] * candidates[:, 3]))
                        index = start + offset
                        joints[index] = result['joints3d'][offset][selected].float().cpu().numpy() / 1000
                        pose[index] = result['pose'][offset][selected].float().cpu().numpy()
                        boxes[index] = candidates[selected].float().cpu().numpy()
                        uncertainty[index] = result['joint_uncertainties'][offset][selected].float().cpu().numpy()
                        valid[index] = True
            dest = root / (sid + '.npz')
            with dest.open('xb') as stream:
                np.savez(stream, joints=joints, pose=pose, box=boxes, uncertainty=uncertainty,
                         valid=valid, timestamps_seconds=times, fps=fps)
            record.update(status='observations_extracted_not_quality_validated',
                          output=str(dest), frames=count, fps=fps, valid_ratio=float(valid.mean()),
                          coordinate_units='meters', timestamps_source='decoder_POS_MSEC',
                          sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
        record['finished_seconds'] = time.time()
        (root / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'records': records,
            'failure_count': sum(x['status'] == 'failed' for x in records)}, indent=2) + '\n')
        print(json.dumps({'sample_id': sid, 'status': record['status'],
                          'valid_ratio': record.get('valid_ratio')}), flush=True)
    if any(x['status'] == 'failed' for x in records):
        raise RuntimeError('Pose extraction failures retained in manifest')


if __name__ == '__main__':
    main()
