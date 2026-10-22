"""Same-condition heldout base/SFT/NFT videos; low-resolution oracle-anchor pilot.
No target video is passed to the sampling pipeline. GT keyframes remain oracle
conditions, so this is not end-to-end deployment quality or music reward proof.
"""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--sft-checkpoint',type=Path,required=True)
    parser.add_argument('--nft-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--ffmpeg',required=True)
    args=parser.parse_args()
    if args.samples<1:parser.error('Positive sample count required')
    for path in (args.upstream,args.model_root,args.metadata,args.output):
        if not path.is_absolute():parser.error('Absolute paths required')
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error('Outputs must be outside publication checkout')
    head=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if head!='b34c83793f783b8bf0c7e82142e06dda01f98d27':raise ValueError('Unexpected upstream revision')
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD']='True'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,str(args.upstream))
    import torch
    from safetensors.torch import load_file
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import ToAbsolutePath
    from diffsynth.utils.data import save_video
    spec=importlib.util.spec_from_file_location('pinned_wan_train',args.upstream/'examples/wanvideo/model_training/train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[str(args.model_root/name) for name in ('local_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')]
    torch.manual_seed(123)
    model=module.WanTrainingModule(model_paths=json.dumps(paths),tokenizer_path=str(args.model_root/'google/umt5-xxl'),
        lora_base_model='dit',lora_target_modules='q,k,v,o,ffn.0,ffn.2',lora_rank=32,device='cpu')
    model.to('cuda');model.pipe.device='cuda';model.eval()
    pipe=model.pipe
    named={name:param for name,param in model.named_parameters() if param.requires_grad}
    weights={'base':{name:param.detach().cpu().clone() for name,param in named.items()},
             'sft':load_file(str(args.sft_checkpoint)), 'nft_oracle':load_file(str(args.nft_checkpoint))}
    for label,state in weights.items():
        if set(state)!=set(named) or any(state[n].shape!=p.shape for n,p in named.items()):
            raise ValueError('Checkpoint key/shape mismatch: '+label)
    data=UnifiedDataset(base_path=str(args.metadata.parent),metadata_path=str(args.metadata),
        data_file_keys=('video','wantodance_reference_image','wantodance_keyframes','wantodance_music_path'),
        main_data_operator=UnifiedDataset.default_video_operator(height=448,width=256,num_frames=149),
        special_operator_map={'wantodance_music_path':ToAbsolutePath(str(args.metadata.parent))})
    report={'status':'running','scope':'256x448_149f_GT_anchor_oracle_heldout_videos',
            'upstream':head,'cfg_scale':1,'sampling_steps':24,'oracle_GT_anchors':True,
            'target_video_passed_to_sampler':False,'quality_approved':False,
            'metadata_sha256':hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            'checkpoint_sha256':{'sft':hashlib.sha256(args.sft_checkpoint.read_bytes()).hexdigest(),
                                 'nft_oracle':hashlib.sha256(args.nft_checkpoint.read_bytes()).hexdigest()},'records':[]}
    receipt=args.output/'report.json';receipt.write_text(json.dumps(report,indent=2))
    try:
        with torch.inference_mode():
            for index in range(min(args.samples,len(data))):
                row=data[index]
                if len(row['wantodance_keyframes'])!=149 or len(row['wantodance_keyframes_mask'])!=149:
                    raise ValueError('149 keyframes and mask required')
                for label,state in weights.items():
                    for n,param in named.items():param.copy_(state[n].to(param))
                    started=time.time();torch.cuda.reset_peak_memory_stats()
                    video=pipe(prompt=row['prompt'],negative_prompt='',seed=60000+index,rand_device='cpu',
                        width=256,height=448,num_frames=149,num_inference_steps=24,cfg_scale=1,sigma_shift=5,tiled=True,
                        wantodance_music_path=row['wantodance_music_path'],
                        wantodance_reference_image=row['wantodance_reference_image'][0],wantodance_fps=30,
                        wantodance_keyframes=row['wantodance_keyframes'],wantodance_keyframes_mask=row['wantodance_keyframes_mask'])
                    if len(video)!=149:raise ValueError('Unexpected generated frame count')
                    silent=args.output/('row'+str(index)+'_'+label+'_silent.mp4')
                    output=args.output/('row'+str(index)+'_'+label+'.mp4')
                    save_video(video,str(silent),fps=30,quality=5)
                    subprocess.run([args.ffmpeg,'-v','error','-n','-i',str(silent),'-i',row['wantodance_music_path'],
                        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t',str(149/30),str(output)],check=True,timeout=120)
                    record={'row':index,'checkpoint':label,'seed':60000+index,'video':str(output),
                            'audio':row['wantodance_music_path'],'reference':str(data.data[index]['wantodance_reference_image']),
                            'GT_video':str(data.data[index]['video']),'frames':149,'fps':30,'width':256,'height':448,
                            'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started,
                            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'quality_approved':False}
                    report['records'].append(record);receipt.write_text(json.dumps(report,indent=2))
                    print('GENERATED',index,label,flush=True)
        report['status']='matched_heldout_videos_completed_not_quality_approved'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:receipt.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
