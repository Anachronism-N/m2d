"""Run fixed Global cases with one official model load; no sampling changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from apply_official_patch import verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--schedule', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    args = parser.parse_args()
    for p in [args.source_root, args.schedule, args.output_root, args.model_root]:
        if not p.is_absolute():
            raise ValueError('All paths must be absolute')
    if args.output_root.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError('Outputs must be outside the publish checkout')
    metadata, _ = verify_source(args.source_root, patched=True)
    schedule_bytes = args.schedule.read_bytes()
    schedule = json.loads(schedule_bytes)
    if schedule.get('schema_version') != 1 or not schedule.get('runs'):
        raise ValueError('Schedule requires nonempty schema_version=1 runs')
    ids = [x['run_id'] for x in schedule['runs']]
    if len(set(ids)) != len(ids) or any('/' in x or x in ['.', '..'] for x in ids):
        raise ValueError('Run IDs must be unique path components')
    manifests = []
    for row in schedule['runs']:
        m = json.loads(Path(row['global_manifest']).read_text())
        if m.get('num_frames') != 149 or m.get('num_inference_steps') != 48:
            raise ValueError('This baseline batch requires official 149 logic frames and 48 steps')
        for k in ['image_path', 'music_path', 'prompt_path']:
            p = Path(m[k])
            if not p.is_absolute() or not p.is_file() or p.stat().st_size == 0:
                raise ValueError('Missing baseline input: ' + k)
        dest = args.output_root / row['run_id'] / 'global'
        if dest.exists():
            raise FileExistsError('Preserve previous run output: ' + str(dest))
        manifests.append((row, m, dest))
    os.environ['WANDANCER_MODEL_ROOT'] = str(args.model_root)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(args.source_root))
    import torch
    import torch.distributed as dist
    from gen_video.gen_video_global import init_dit_model, get_music_base_feature, gen_video
    # Native initialization establishes the exact eight-rank USP topology.
    pipe = init_dit_model(SimpleNamespace())
    rank = dist.get_rank()
    for row, m, dest in manifests:
        started = time.time()
        if rank == 0:
            dest.mkdir(parents=True, exist_ok=False)
            (dest / 'videos').mkdir()
        dist.barrier()
        feature_path = dest / 'native35.npy'
        output = dest / 'videos' / (Path(m['image_path']).stem + '_' + Path(m['music_path']).stem + '_seed' + str(m['seed']) + '_' + row['run_id'] + '.mp4')
        receipt = {'run_id': row['run_id'], 'stage': 'global', 'execution_status': 'RUNNING',
                   'quality_status': 'NOT_EVALUATED', 'official_commit': metadata['official_commit'],
                   'patch_sha256': metadata['patch_sha256'], 'manifest': m,
                   'schedule_sha256': hashlib.sha256(schedule_bytes).hexdigest(),
                   'expected_output': str(output), 'started_unix_seconds': started,
                   'native_pipeline_model_reused': True}
        if rank == 0:
            (dest / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            get_music_base_feature(m['music_path'], str(feature_path), fps=30)
        dist.barrier(device_ids=[rank])
        torch.cuda.reset_peak_memory_stats()
        prompt = Path(m['prompt_path']).read_text().strip()
        try:
            gen_video(pipe, str(feature_path), m['image_path'], prompt, str(output),
                seed=m['seed'], height=m['height'], width=m['width'], num_frames=149,
                enable_refimage=True, refimage_path=m['image_path'], enable_global=True,
                enable_dynamicfps=True, enable_vae_decode_framewise=True,
                enable_skip_layer=True, enable_unimodel=True, sigma_shift=m['sigma_shift'],
                num_inference_steps=48, cfg_scale=m['cfg_scale'])
            dist.barrier(device_ids=[rank])
            if rank == 0:
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError('Missing generated container')
                receipt.update(execution_status='GENERATED_NOT_QUALITY_VALIDATED',
                               returncode=0, finished_unix_seconds=time.time(),
                               output_size_bytes=output.stat().st_size)
                (dest / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        except BaseException as error:
            if rank == 0:
                receipt.update(execution_status='FAILED', error_type=type(error).__name__, finished_unix_seconds=time.time())
                (dest / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            raise
        (dest / ('resources_rank' + str(rank) + '.json')).write_text(json.dumps({
            'rank': rank, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'elapsed_seconds': time.time() - started}) + '\n')
        dist.barrier(device_ids=[rank])
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
