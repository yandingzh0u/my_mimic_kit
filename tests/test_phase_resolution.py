"""Statistical support, rather than control frequency, limits phase strata."""
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mimickit'))
from learning.phase_sampler import PhaseSampler, supported_resolution
from test_phase_sampler import _agent
from learning.ppo_agent import PPOAgent


def brute(phase,frames,budgets,masks):
    pools=[(phase.flatten().double() if mask is None else phase.flatten().double()[mask.flatten()],b)
           for b,mask in zip(budgets,masks)]
    pools=[(p,min(b,len(p))) for p,b in pools if len(p)]
    upper=min(frames,len(phase),*(b for _,b in pools))
    for k in range(upper,0,-1):
        if all(int(torch.bincount((p*k).long().clamp_max(k-1),minlength=k).min())>=math.ceil(b/k)
               for p,b in pools): return k


def test_largest_feasible_resolution_against_exhaustive_search():
    rng=np.random.default_rng(721)
    gen=torch.Generator().manual_seed(721)
    for case in range(600):
        n=int(rng.integers(1,193));frames=[1,8,299,3000,10**9][case%5]
        phase=torch.rand(n,dtype=torch.float64,generator=gen)
        if case%4==0: phase=phase**8
        if case%4==1: phase[phase>.3]=1
        mask=torch.rand(n,generator=gen)>.3
        a,b=int(rng.integers(1,257)),int(rng.integers(1,257))
        assert supported_resolution(phase,frames,[a,b],[mask,None])==brute(phase,frames,[a,b],[mask,None])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_boundary_values_do_not_change_maximum_feasible_result(dtype):
    for denominator in range(2,23):
        edges=torch.arange(denominator+1,dtype=dtype)/denominator
        phase=torch.cat([edges,torch.nextafter(edges,torch.zeros_like(edges)),torch.nextafter(edges,torch.ones_like(edges))])
        for b in [3,9,len(phase)//2,len(phase),2*len(phase)]:
            assert supported_resolution(phase,299,[b],[None])==brute(phase,299,[b],[None])


def test_joint_actor_support_can_force_coarser_resolution():
    phase=(torch.arange(800,dtype=torch.float64)+.5)/800
    mask=phase>.7
    assert supported_resolution(phase,299,[100,100],[mask,None])==1
    assert supported_resolution(phase,299,[100],[None])>1


def test_no_empty_bins_and_population_weight_bound():
    phase=torch.linspace(0,1,1024,dtype=torch.float64)**4
    k=supported_resolution(phase,299,[128,64],[None,None])
    sampler=PhaseSampler(phase,k)
    assert sampler.num_present==k
    assert int(sampler.counts.min())>=math.ceil(128/k)
    assert float(sampler.full_weights().max())<=1/128+1e-14
    for _ in range(30):
        idx,w,quota=sampler.sample(128)
        assert len(idx.unique())==128 and bool((quota<=sampler.counts).all())
        assert int(quota.max())-int(quota.min())<=1


def test_empty_actor_does_not_disable_critic_partition():
    phase=torch.linspace(0,1,256)
    assert supported_resolution(phase,299,[128,64],[torch.zeros(256,dtype=torch.bool),None])==supported_resolution(phase,299,[64],[None])


def test_ordinary_ppo_uses_uniform_advantage_statistics():
    agent=_agent(agent_type=PPOAgent)
    agent._exp_buffer.get_data('diag_phase').zero_()
    with torch.no_grad():
        info=agent._build_train_data()
        raw=agent._exp_buffer.get_data_flat('tar_val')-agent._model.eval_critic(
            agent._obs_norm.normalize(agent._exp_buffer.get_data_flat('obs'))).squeeze(-1)
        eligible=agent._exp_buffer.get_data_flat('rand_action_mask')==1
        torch.testing.assert_close(info['adv_mean'],raw[eligible].mean())
    assert torch.equal(agent._exp_buffer.get_data('phase_weight'),torch.ones(4,8))
    with mock.patch.object(agent._exp_buffer,'sample',return_value={}) as draw:
        assert agent._sample_ppo_batch(16,actor=True)=={}
        draw.assert_called_once_with(16)


def test_supported_sampling_integrates_with_actual_ppo_updates():
    agent=_agent()
    with torch.no_grad(): info=agent._build_train_data()
    for sampler,b in [(agent._actor_phase_sampler,32),(agent._critic_phase_sampler,16)]:
        assert sampler.num_present==sampler.num_bins
        assert int(sampler.counts.min())>=math.ceil(min(b,len(sampler.indices))/sampler.num_bins)
    assert info['phase_motion_control_cells']==299
    assert info['phase_resolution_bins']<299
    update=agent._update_model()
    assert all(bool(torch.isfinite(v).all()) for v in update.values())
