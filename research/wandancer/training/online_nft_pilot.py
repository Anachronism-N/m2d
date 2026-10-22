"""Bounded on-policy NFT oracle-latent alignment pilot, not music/quality RL.

GT endpoint proximity is an explicit oracle reward for engineering validation.
It is not a unique choreography target, deployable reward, or human preference.
All rollouts start from Gaussian noise; GT is used only for reward and the
provided sparse training-anchor condition. CFG=1 for every scoring/update branch.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nft_objective import nft_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--group-size',type=int,default=2)
    p.add_argument('--sampling-steps',type=int,default=24)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--height',type=int,default=448)
    p.add_argument('--frames',type=int,default=149)
    a=p.parse_args()
    if a.rounds<1 or a.group_size<2 or a.sampling_steps<2:
        p.error('positive rounds, group>=2 and sampling steps>=2 required')
    if a.frames%4!=1 or a.width%16 or a.height%16 or ((a.width//16)*(a.height//16))%8:
        p.error('native temporal and eight-group spatial layout required')
    if a.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        p.error('output must be outside publication checkout')
    head=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if head!='b34c83793f783b8bf0c7e82142e06dda01f98d27':raise ValueError('Unexpected upstream')
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD']='True'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(a.upstream))
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import ToAbsolutePath
    from safetensors.torch import save_file,load_file
    a.output.mkdir(parents=True,exist_ok=False)
    record={'status':'initializing','experiment':'online_NFT_oracle_latent_engineering_pilot',
            'music_reward_validated':False,'human_quality_verified':False,
            'oracle_reward':'negative_mean_squared_clean_latent_distance_to_training_GT',
            'cfg_scale':1,'old_policy_update':'on-policy hard synchronization each round, no EMA',
            'geometry':[a.width,a.height,a.frames],'upstream_commit':head,'rounds':[],
            'metadata_sha256':hashlib.sha256(a.metadata.read_bytes()).hexdigest()}
    receipt=a.output/'receipt.json'
    receipt.write_text(json.dumps(record,indent=2))
    path=a.upstream/'examples/wanvideo/model_training/train.py'
    spec=importlib.util.spec_from_file_location('pinned_wan_training',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=UnifiedDataset(base_path=str(a.metadata.parent),metadata_path=str(a.metadata),
        data_file_keys=('video','wantodance_reference_image','wantodance_keyframes','wantodance_music_path'),
        main_data_operator=UnifiedDataset.default_video_operator(base_path=str(a.metadata.parent),
            height=a.height,width=a.width,num_frames=a.frames,height_division_factor=16,width_division_factor=16),
        special_operator_map={'wantodance_music_path':ToAbsolutePath(str(a.metadata.parent))})
    paths=[str(a.model_root/n) for n in ('local_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    started=time.time()
    model=module.WanTrainingModule(model_paths=json.dumps(paths),tokenizer_path=str(a.model_root/'google/umt5-xxl'),
        lora_base_model='dit',lora_target_modules='q,k,v,o,ffn.0,ffn.2',lora_rank=32,
        extra_inputs='wantodance_music_path,wantodance_reference_image,wantodance_fps,wantodance_keyframes,wantodance_keyframes_mask',
        use_gradient_checkpointing=True,use_gradient_checkpointing_offload=True,device='cpu')
    model.to('cuda');model.pipe.device='cuda';model.eval()
    pipe=model.pipe
    named={n:x for n,x in model.named_parameters() if x.requires_grad}
    if not named or any('lora_' not in n for n in named):raise RuntimeError('Unexpected trainable parameters')
    optimizer=torch.optim.AdamW(named.values(),lr=1e-5)
    layers=[x for x in pipe.dit.modules() if isinstance(x,BaseTunerLayer)]
    reference={n:x.detach().cpu().clone() for n,x in named.items()}
    models={n:getattr(pipe,n) for n in pipe.in_iteration_models}
    def snapshot():return {n:x.detach().cpu().clone() for n,x in named.items()}
    def restore(weights):
        with torch.no_grad():
            for n,x in named.items():x.copy_(weights[n].to(x))
    def prepare(row):
        sample=data[row];sample['wantodance_keyframes_mask']=sample['wantodance_keyframes_mask'][:a.frames]
        inputs=model.get_pipeline_inputs(sample)
        inputs=model.transfer_data_to_device(inputs,'cuda',pipe.torch_dtype)
        with torch.no_grad():
            for unit in pipe.units:inputs=pipe.unit_runner(unit,pipe,*inputs)
        shared,positive,negative=inputs
        target=shared['input_latents'].detach()
        kwargs={**shared,**positive}
        kwargs.pop('latents',None);kwargs.pop('timestep',None)
        return kwargs,target
    def velocity(state,sigma,kwargs,gradient):
        kwargs=dict(kwargs);kwargs['latents']=state.to(pipe.torch_dtype)
        kwargs['use_gradient_checkpointing']=gradient
        kwargs['use_gradient_checkpointing_offload']=gradient
        timestep=torch.tensor([sigma*1000],device='cuda',dtype=pipe.torch_dtype)
        return pipe.model_fn(**models,**kwargs,timestep=timestep)
    def rollout(seed,kwargs,target):
        pipe.scheduler.set_timesteps(a.sampling_steps,shift=5)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        state=torch.randn(target.shape,device='cuda',dtype=pipe.torch_dtype,generator=generator)
        with torch.no_grad():
            for timestep in pipe.scheduler.timesteps:
                sigma=float(timestep)/1000
                pred=velocity(state,sigma,kwargs,False)
                state=pipe.scheduler.step(pred,timestep,state)
        if not torch.isfinite(state).all():raise RuntimeError('Nonfinite rollout endpoint')
        reward=-float((state.float()-target.float()).square().mean())
        return state.detach(),reward
    try:
        torch.cuda.reset_peak_memory_stats()
        for round_id in range(a.rounds):
            pipe.scheduler.set_timesteps(1000,training=True)
            kwargs,target=prepare(round_id%min(len(data),2))
            old=snapshot()
            endpoints=[];rewards=[]
            seeds=[50000+round_id*100+i for i in range(a.group_size)]
            for seed in seeds:
                endpoint,reward=rollout(seed,kwargs,target);endpoints.append(endpoint.cpu());rewards.append(reward)
            values=torch.tensor(rewards,dtype=torch.float32)
            advantages=(values-values.mean())/(values.std(unbiased=False)+1e-4)
            entry={'round':round_id+1,'row':round_id%min(len(data),2),'seeds':seeds,
                   'rewards':rewards,'advantages':advantages.tolist(),'same_context':True,'updates':[]}
            record['rounds'].append(entry);record['status']='updating';receipt.write_text(json.dumps(record,indent=2))
            for i,x0_cpu in enumerate(endpoints):
                x0=x0_cpu.to('cuda')
                sigma=torch.tensor([.5],device='cuda')
                noise=torch.randn_like(x0)
                xt=(1-sigma.view(1,1,1,1,1))*x0+sigma.view(1,1,1,1,1)*noise
                current_weights=snapshot();restore(old)
                with torch.no_grad():old_velocity=velocity(xt,.5,kwargs,False).detach()
                for layer in layers:layer.enable_adapters(False)
                try:
                    with torch.no_grad():ref_velocity=velocity(xt,.5,kwargs,False).detach()
                finally:
                    for layer in layers:layer.enable_adapters(True)
                restore(current_weights)
                optimizer.zero_grad(set_to_none=True)
                current=velocity(xt,.5,kwargs,True)
                loss,metrics=nft_loss(xt.float(),x0.float(),sigma,current.float(),old_velocity.float(),ref_velocity.float(),advantages[i:i+1].to('cuda'))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite NFT loss')
                loss.backward()
                grads=[x.grad for x in named.values() if x.grad is not None]
                if not grads or not all(torch.isfinite(x).all() for x in grads):raise RuntimeError('Invalid NFT gradients')
                norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1)
                optimizer.step()
                entry['updates'].append({'candidate':i,'loss':float(loss.detach()),'gradient_norm':float(norm),
                                         'gradient_tensor_count':len(grads),'policy_loss':float(metrics['policy'])})
                receipt.write_text(json.dumps(record,indent=2))
            post=[]
            for seed in seeds:
                endpoint,reward=rollout(seed,kwargs,target);post.append(reward)
            entry['post_update_same_seed_rewards']=post
            entry['mean_reward_delta']=sum(post)/len(post)-sum(rewards)/len(rewards)
            receipt.write_text(json.dumps(record,indent=2))
        checkpoint=snapshot()
        changed=sum(not torch.equal(checkpoint[n],reference[n]) for n in named)
        if not changed:raise RuntimeError('NFT never changed any adapter tensor')
        save_file({n:x.contiguous() for n,x in checkpoint.items()},str(a.output/'lora.safetensors'))
        saved=load_file(str(a.output/'lora.safetensors'))
        if set(saved)!=set(checkpoint) or not all(torch.equal(saved[n],checkpoint[n]) for n in checkpoint):
            raise RuntimeError('NFT checkpoint exact reload failed')
        torch.save({'optimizer':optimizer.state_dict(),'round':a.rounds,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},a.output/'training_state.pt')
        record.update(status='online_nft_oracle_engineering_pilot_completed',optimizer_steps=a.rounds*a.group_size,
                      changed_trainable_tensors=changed,checkpoint_exact_reload=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      elapsed_seconds=time.time()-started,quality_verified=False,
                      warning='Oracle training-endpoint reward; not deployment music alignment or human quality RL')
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:receipt.write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
