"""Real Local LoRA gradient/optimizer/checkpoint smoke; not a quality experiment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--frames', type=int, default=149)
    p.add_argument('--steps', type=int, default=2)
    a = p.parse_args()
    if a.steps < 1 or a.frames % 4 != 1 or a.width % 16 or a.height % 16:
        p.error('positive steps, 4n+1 frames and spatial multiples of16 required')
    # Native WanToDance divides each latent spatial grid into eight music groups.
    if ((a.height // 16) * (a.width // 16)) % 8:
        p.error('native single-device music grouping requires latent spatial tokens divisible by8')
    if a.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        p.error('output must be private outside checkout')
    a.output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if head != 'b34c83793f783b8bf0c7e82142e06dda01f98d27':
        raise ValueError('Unexpected upstream revision')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(a.upstream))
    import torch
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import ToAbsolutePath
    from safetensors.torch import save_file,load_file
    path=a.upstream/'examples/wanvideo/model_training/train.py'
    spec=importlib.util.spec_from_file_location('wan_train_pinned',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=UnifiedDataset(base_path=str(a.metadata.parent),metadata_path=str(a.metadata),
        data_file_keys=('video','wantodance_reference_image','wantodance_keyframes','wantodance_music_path'),
        main_data_operator=UnifiedDataset.default_video_operator(base_path=str(a.metadata.parent),height=a.height,width=a.width,num_frames=a.frames,height_division_factor=16,width_division_factor=16),
        special_operator_map={'wantodance_music_path':ToAbsolutePath(str(a.metadata.parent))})
    models=[str(a.model_root/n) for n in ('local_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    start=time.time()
    model=module.WanTrainingModule(model_paths=json.dumps(models),tokenizer_path=str(a.model_root/'google/umt5-xxl'),lora_base_model='dit',lora_target_modules='q,k,v,o,ffn.0,ffn.2',lora_rank=32,extra_inputs='wantodance_music_path,wantodance_reference_image,wantodance_fps,wantodance_keyframes,wantodance_keyframes_mask',use_gradient_checkpointing=True,use_gradient_checkpointing_offload=True,device='cpu')
    model.to('cuda');model.pipe.device='cuda'
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not named or any('lora_' not in n for n in named):raise RuntimeError('Unexpected trainable parameter contract')
    optimizer=torch.optim.AdamW(named.values(),lr=1e-4)
    sample=data[0]
    if len(sample['video'])!=a.frames:raise ValueError('Frame count mismatch')
    sample['wantodance_keyframes_mask']=sample['wantodance_keyframes_mask'][:a.frames]
    before={n:p.detach().cpu().clone() for n,p in named.items()}
    torch.cuda.reset_peak_memory_stats()
    record={'status':'running','smoke_only':True,'upstream_commit':head,'geometry':[a.width,a.height,a.frames],'lora_rank':32,'trainable_parameters':sum(p.numel() for p in named.values()),'metrics':[]}
    (a.output/'receipt.json').write_text(json.dumps(record,indent=2))
    try:
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            loss=model(sample)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            gradients=[p.grad for p in named.values() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise RuntimeError('Missing/nonfinite gradients')
            norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1)
            optimizer.step()
            record['metrics'].append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm),'parameters_with_grad':len(gradients)})
            (a.output/'receipt.json').write_text(json.dumps(record,indent=2))
        changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in named.items())
        if not changed:raise RuntimeError('Optimizer did not change any trainable tensor')
        checkpoint={n:p.detach().cpu().contiguous() for n,p in named.items()}
        save_file(checkpoint,str(a.output/'lora.safetensors'))
        torch.save({'optimizer':optimizer.state_dict(),'step':a.steps,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},a.output/'training_state.pt')
        restored=load_file(str(a.output/'lora.safetensors'))
        if set(restored)!=set(named) or not all(torch.equal(restored[n],p.detach().cpu()) for n,p in named.items()):raise RuntimeError('Checkpoint values differ')
        with torch.no_grad():
            first=next(iter(named));named[first].zero_();named[first].copy_(restored[first].to(named[first]))
        restored_optimizer=torch.optim.AdamW(named.values(),lr=1e-4)
        saved=torch.load(a.output/'training_state.pt',map_location='cpu',weights_only=False)
        restored_optimizer.load_state_dict(saved['optimizer'])
        if saved['step']!=a.steps or len(restored_optimizer.state)!=len(optimizer.state):raise RuntimeError('Optimizer restore failed')
        record.update(status='gradient_optimizer_checkpoint_smoke_passed',changed_trainable_tensors=changed,checkpoint_exact=True,optimizer_restored=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.time()-start,quality_verified=False,full_training_resume_trajectory_verified=False)
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:(a.output/'receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
