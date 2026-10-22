"""Common heldout flow-MSE comparison at nonzero fixed sigmas; not human quality."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--sft-checkpoint',type=Path,required=True)
    p.add_argument('--nft-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=13)
    a=p.parse_args()
    if a.samples<1:p.error('Positive sample count required')
    if a.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):p.error('Private output required')
    head=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if head!='b34c83793f783b8bf0c7e82142e06dda01f98d27':raise ValueError('Unexpected upstream revision')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD']='True'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(a.upstream))
    import torch
    from safetensors.torch import load_file
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import ToAbsolutePath
    path=a.upstream/'examples/wanvideo/model_training/train.py'
    spec=importlib.util.spec_from_file_location('wan_compare_source',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    models=[str(a.model_root/n) for n in ('local_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    model=module.WanTrainingModule(model_paths=json.dumps(models),tokenizer_path=str(a.model_root/'google/umt5-xxl'),
        lora_base_model='dit',lora_target_modules='q,k,v,o,ffn.0,ffn.2',lora_rank=32,
        extra_inputs='wantodance_music_path,wantodance_reference_image,wantodance_fps,wantodance_keyframes,wantodance_keyframes_mask',
        device='cpu',use_gradient_checkpointing=True,use_gradient_checkpointing_offload=True)
    model.to('cuda');model.pipe.device='cuda';model.eval()
    named={n:x for n,x in model.named_parameters() if x.requires_grad}
    weights={'base':{n:x.detach().cpu().clone() for n,x in named.items()},
             'sft':load_file(str(a.sft_checkpoint)),'nft_oracle':load_file(str(a.nft_checkpoint))}
    for name,state in weights.items():
        if set(state)!=set(named):raise ValueError('Checkpoint trainable key mismatch: '+name)
    data=UnifiedDataset(base_path=str(a.metadata.parent),metadata_path=str(a.metadata),
        data_file_keys=('video','wantodance_reference_image','wantodance_keyframes','wantodance_music_path'),
        main_data_operator=UnifiedDataset.default_video_operator(height=448,width=256,num_frames=149),
        special_operator_map={'wantodance_music_path':ToAbsolutePath(str(a.metadata.parent))})
    pipe=model.pipe; forward_models={n:getattr(pipe,n) for n in pipe.in_iteration_models}
    report={'status':'running','geometry':[256,448,149],'cfg_scale':1,'sigmas':[.25,.5,.75],
            'upstream_commit':head,'timestep_dtype':str(pipe.torch_dtype),
            'model_timesteps':[float(torch.tensor(s*1000,dtype=pipe.torch_dtype)) for s in [.25,.5,.75]],
            'sigma_contract':'nominal noise sigmas; model timestep follows native BF16 quantization',
            'reward':'none, raw flow-target MSE only','oracle_GT_anchors':True,
            'metadata_sha256':hashlib.sha256(a.metadata.read_bytes()).hexdigest(),
            'checkpoint_hashes':{k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('sft',a.sft_checkpoint),('nft_oracle',a.nft_checkpoint)]},
            'split_identity_music_isolation_verified':False,'samples':[], 'quality_verified':False}
    receipt=a.output/'report.json'
    receipt.write_text(json.dumps(report,indent=2))
    started=time.time()
    with torch.inference_mode():
        for i in range(min(a.samples,len(data))):
            pipe.scheduler.set_timesteps(1000,training=True)
            sample=data[i]
            if len(sample['video'])!=149 or len(sample['wantodance_keyframes_mask'])!=149:
                raise ValueError('Unexpected temporal shape')
            inputs=model.get_pipeline_inputs(sample)
            inputs=model.transfer_data_to_device(inputs,'cuda',pipe.torch_dtype)
            for unit in pipe.units:inputs=pipe.unit_runner(unit,pipe,*inputs)
            shared,positive,_=inputs; target=shared['input_latents'].detach()
            torch.manual_seed(90000+i);torch.cuda.manual_seed_all(90000+i)
            noise=torch.randn_like(target)
            flow_target=noise-target
            kwargs={**shared,**positive}
            kwargs['use_gradient_checkpointing']=False
            kwargs['use_gradient_checkpointing_offload']=False
            row={'index':i,'noise_seed':90000+i,'scores':{}}
            for name,state in weights.items():
                for n,x in named.items():x.copy_(state[n].to(x))
                losses=[]
                for sigma in report['sigmas']:
                    kwargs['latents']=((1-sigma)*target+sigma*noise).to(pipe.torch_dtype)
                    timestep=torch.tensor([sigma*1000],dtype=pipe.torch_dtype,device='cuda')
                    prediction=pipe.model_fn(**forward_models,**kwargs,timestep=timestep)
                    loss=torch.nn.functional.mse_loss(prediction.float(),flow_target.float())
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite raw evaluation loss')
                    losses.append(float(loss))
                row['scores'][name]=losses
            report['samples'].append(row)
            receipt.write_text(json.dumps(report,indent=2))
            print('COMPARED_HELDOUT',i,flush=True)
    means={name:sum(sum(x['scores'][name]) for x in report['samples'])/(3*len(report['samples'])) for name in weights}
    report.update(status='common_heldout_raw_flow_mse_completed',means=means,elapsed_seconds=time.time()-started,
                  warning='Different training budgets and objectives; flow MSE is not reward improvement or visual quality')
    receipt.write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'samples':len(report['samples']),'means':means}),flush=True)


if __name__=='__main__':main()
