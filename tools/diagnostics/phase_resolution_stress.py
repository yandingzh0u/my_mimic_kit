"""Support-limited resolution stress, including holes and long motions."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mimickit'))
from learning.phase_resolution import supported_resolution
from learning.phase_sampler import PhaseSampler


def cpu(cases,seed):
    rng=np.random.default_rng(seed);gen=torch.Generator().manual_seed(seed)
    draws=batches=0;minimum_k=10**9;maximum_k=0
    for case in range(cases):
        n=int(rng.integers(1,1025));f=[1,8,299,3000,10**9][case%5]
        phase=torch.rand(n,dtype=torch.float64,generator=gen)
        if case%5==0: phase=phase**8
        if case%5==1: phase=torch.where(phase<.5,phase*.2,.8+phase*.2)
        if case%5==2: phase[:-1]=0;phase[-1]=1
        if case%5==3: phase=(phase*17).long().double()/17
        eligible=torch.rand(n,generator=gen)>.3
        if case%17==0: eligible.zero_()
        ba=int(rng.integers(1,max(2,n*2)));bc=max(1,ba//2)
        k=supported_resolution(phase,f,[ba,bc],[eligible,None])
        minimum_k=min(minimum_k,k);maximum_k=max(maximum_k,k)
        for mask,b in [(eligible,ba),(None,bc)]:
            sampler=PhaseSampler(phase,k,min(ba,bc),mask)
            if len(sampler.indices):
                q=math.ceil(min(b,len(sampler.indices))/k)
                assert len(sampler.present_bins)==k and int(sampler.counts.min())>=q
                assert float(sampler.full_weights().max())<=1/min(b,len(sampler.indices))+1e-14
            for _ in range(2):
                idx,w,quota=sampler.sample(b,generator=gen)
                assert len(idx.unique())==len(idx)==min(b,len(sampler.indices))
                assert bool(((quota>=1)&(quota<=sampler.counts)).all())
                mass=torch.zeros(sampler.num_bins,dtype=torch.float64).scatter_add_(0,sampler.bin_ids[idx],w)
                torch.testing.assert_close(mass[sampler.present_bins],sampler.target_mass,atol=1e-12,rtol=0)
                if len(idx): assert float(w.max())<=2/len(idx)+1e-14
                draws+=len(idx);batches+=1
        if (case+1)%5000==0: print(json.dumps(dict(distributions=case+1,batches=batches)),flush=True)
    return dict(distributions=cases,minibatches=batches,sampled_indices=draws,selected_bins_range=[minimum_k,maximum_k],support_violations=0,empty_bin_violations=0,duplicate_violations=0,weight_violations=0)


def gpu(seed):
    torch.manual_seed(seed)
    n=262144;p=(torch.arange(n,device='cuda:0',dtype=torch.float64)+.5)/n
    profiles={'uniform':p,'early_sparse':p.sqrt(),
              'interior_hole':torch.where(p<.5,p*.2,.8+p*.2),
              'interior_weak':torch.cat((p[:130572]*.2,
                  (torch.arange(1000,device=p.device,dtype=p.dtype)+.5)/1000*.8+.1,
                  p[:130572]*.2+.9)).clamp_max(1)}
    result={}
    for name,phase in profiles.items():
        for f in [299,10**9]:
            torch.cuda.synchronize();start=time.perf_counter()
            k=supported_resolution(phase,f,[32768,16384],[None,None])
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
            sampler=PhaseSampler(phase,k,16384)
            for b in [16384,32768]:
                for _ in range(20):
                    idx,w,q=sampler.sample(b)
                    assert len(idx.unique())==b and bool((q<=sampler.counts).all()) and bool((q>=1).all())
                    assert float(sampler.full_weights().max())<=1/32768+1e-14
            result[f'{name}:{f}']=dict(selected_bins=k,resolution_ms=elapsed,minimum_pool=int(sampler.counts.min()))
            print(json.dumps(dict(profile=name,frames=f,**result[f'{name}:{f}'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=25000)
    parser.add_argument('--gpu-only',action='store_true')
    parser.add_argument('--seed',type=int,default=20261025)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if args.gpu_only:
        result=gpu(args.seed);name='support_gpu_stress.json'
    else:
        result=cpu(args.cases,args.seed);name='support_stress.json'
    (args.out/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
