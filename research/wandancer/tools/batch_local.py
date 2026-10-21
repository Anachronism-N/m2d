"""Run paired native Local cases with one model load and explicit source validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from apply_official_patch import verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=21600)
    args = parser.parse_args()
    for path in (args.source_root, args.schedule, args.model_root, args.output_root):
        if not path.is_absolute():
            raise ValueError('Paths must be absolute')
    source = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    publication_root = Path(__file__).resolve().parents[3]
    if output_root.is_relative_to(publication_root) or output_root.is_relative_to(source):
        raise ValueError('Outputs must be outside code checkouts')
    metadata, _ = verify_source(source, patched=True)
    payload = args.schedule.read_bytes()
    schedule = json.loads(payload)
    rows = schedule.get('runs')
    if schedule.get('schema_version') != 1 or not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty versioned run schedule required')
    ids = [row['run_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Run IDs must be unique')
    manifests = []
    for row in rows:
        run_id = row['run_id']
        if not isinstance(run_id, str) or not run_id or any(x in run_id for x in ('/', '\\', '\n', '\r')) or run_id in ('.', '..'):
            raise ValueError('Safe run path component required')
        manifest = json.loads(Path(row['manifest']).read_text())
        for key in ('image_path', 'music_path', 'prompt_path'):
            path = Path(manifest[key])
            if not path.is_absolute() or not path.is_file():
                raise ValueError('Missing required input: ' + key)
        if manifest.get('num_frames') != 149 or manifest.get('num_inference_steps') != 24:
            raise ValueError('Native Local baseline requires 149 frames and 24 steps')
        dest = output_root / run_id / 'local'
        if dest.exists():
            raise FileExistsError('Prior run is preserved: ' + str(dest))
        manifests.append((row, manifest, dest))
    os.environ['WANDANCER_MODEL_ROOT'] = str(args.model_root)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(source))
    import torch
    import torch.distributed as dist
    from gen_video import gen_video_local as native
    pipe = native.init_dit_model(argparse.Namespace())
    rank = dist.get_rank()
    # Native process still constructs all conditions, slices audio and saves video.
    # Reuse only the already-loaded model; do not change denoising or conditions.
    native.init_dit_model = lambda unused_args: pipe
    for row, manifest, dest in manifests:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            state = [None]
            if rank == 0:
                receipt_path = Path(row['global_receipt'])
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    state[0] = receipt.get('execution_status')
                else:
                    state[0] = 'PENDING'
            dist.broadcast_object_list(state, src=0)
            if state[0] == 'FAILED':
                raise RuntimeError('Global failed; refuse invalid plan')
            if state[0] == 'GENERATED_NOT_QUALITY_VALIDATED':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Global plan not ready within bounded wait')
            time.sleep(15)
        global_path = Path(manifest['global_video_path'])
        if not global_path.is_absolute() or not global_path.is_file():
            raise ValueError('Missing real generated Global plan')
        started = time.time()
        if rank == 0:
            dest.mkdir(parents=True, exist_ok=False)
            (dest / 'videos').mkdir()
        dist.barrier()
        name = Path(manifest['image_path']).stem + '_' + Path(manifest['music_path']).stem
        output = dest / 'videos' / (name + '_' + row['run_id'] + '_seed' + str(manifest['seed']) + '.mp4')
        record = {'run_id': row['run_id'], 'stage': 'local', 'manifest': manifest,
                  'expected_output': str(output), 'execution_status': 'RUNNING',
                  'quality_status': 'NOT_EVALUATED', 'native_model_reused': True,
                  'official_commit': metadata['official_commit'],
                  'patch_sha256': metadata['patch_sha256'],
                  'schedule_sha256': hashlib.sha256(payload).hexdigest(),
                  'started_unix_seconds': started}
        receipt = dest / 'run_receipt.json'
        if rank == 0:
            receipt.write_text(json.dumps(record, indent=2) + '\n')
        torch.cuda.reset_peak_memory_stats()
        previous_cwd, previous_argv = Path.cwd(), sys.argv
        os.chdir(dest)
        sys.argv = ['native-local']
        for key, value in manifest.items():
            sys.argv.extend(['--' + key, str(value)])
        sys.argv.extend(['--timestamp', row['run_id'], '--output_folder', str(dest / 'videos')])
        try:
            native.process()
            peaks = [None] * dist.get_world_size()
            dist.all_gather_object(peaks, {'rank': rank, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                         'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            if rank == 0:
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError('Local expected output missing')
                record.update(execution_status='GENERATED_NOT_QUALITY_VALIDATED', returncode=0,
                              finished_unix_seconds=time.time(), output_size_bytes=output.stat().st_size,
                              gpu_peaks=peaks)
                receipt.write_text(json.dumps(record, indent=2) + '\n')
                print('LOCAL_BATCH_GENERATED', row['run_id'], flush=True)
        except BaseException as error:
            if rank == 0:
                record.update(execution_status='FAILED', error_type=type(error).__name__)
                receipt.write_text(json.dumps(record, indent=2) + '\n')
            raise
        finally:
            os.chdir(previous_cwd)
            sys.argv = previous_argv
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
