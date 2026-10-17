"""Mathematical contracts for the isolated temporal discriminator experiment."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/diagnostics'))
from temporal_dare_stress import SQRT2, TemporalReward, collect_training, lift, orthogonal_to_concat, trace_pairs


def test_orthogonal_energy_and_current_reconstruction():
    torch.manual_seed(927)
    pair=torch.randn(64,2,172,dtype=torch.float64)
    current,previous=pair.unbind(1)
    average=(current+previous)/SQRT2
    change=(current-previous)/SQRT2
    torch.testing.assert_close((average.square()+change.square()).sum(-1),
                               pair.square().sum((-2,-1)))
    torch.testing.assert_close((average+change)/SQRT2,current)


def test_function_preservation_with_spectral_state_and_input_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(928)
    # Input coordinates are intentionally sparse; the complete model remains
    # the production-shaped CPL trunk, rather than a substitute MLP.
    groups=[('root_pos',(0,1,2)),('dof_vel',(144,145,146))]
    original=TemporalReward(groups).double().eval()
    pair=torch.randn(12,2,172,dtype=torch.float64,requires_grad=True)
    expected=original.logits(pair)
    gradient=torch.autograd.grad(expected.sum(),pair)[0]
    for basis in ('concat','orthogonal'):
        expanded=lift(original,basis).eval()
        x=pair.detach().clone().requires_grad_(True)
        actual=expanded.logits(x)
        actual_gradient=torch.autograd.grad(actual.sum(),x)[0]
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
        torch.testing.assert_close(actual_gradient,gradient,atol=1e-12,rtol=1e-10)
        assert actual_gradient[:,1].abs().max()==0


def test_sgd_momentum_weight_decay_are_rotation_equivariant():
    torch.manual_seed(929)
    base=torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(6,4)).double()
    rotated=torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(6,4)).double()
    with torch.no_grad():
        rotated.parametrizations.weight.original.copy_(orthogonal_to_concat(base.parametrizations.weight.original))
        rotated.parametrizations.weight[0]._u.copy_(base.parametrizations.weight[0]._u)
        rotated.parametrizations.weight[0]._v.copy_(orthogonal_to_concat(base.parametrizations.weight[0]._v))
        rotated.bias.copy_(base.bias)
    optimizers=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.03)
                for m in (base,rotated)]
    for _ in range(12):
        x=torch.randn(24,6,dtype=torch.float64)
        target=torch.randn(24,4,dtype=torch.float64)
        for model,opt,inputs in zip((base,rotated),optimizers,(x,orthogonal_to_concat(x))):
            opt.zero_grad()
            (model(inputs)-target).square().mean().backward()
            opt.step()
    base.eval();rotated.eval()
    torch.testing.assert_close(base(x),rotated(orthogonal_to_concat(x)),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(base.parametrizations.weight.original,
        orthogonal_to_concat(rotated.parametrizations.weight.original),atol=1e-12,rtol=1e-10)


def test_trace_history_uses_reset_frame_then_previous_observed_frame():
    trace={'norm_diff':np.arange(3*2*172).reshape(3,2,172),
           'initial_previous':np.full((2,172),-17)}
    pair=trace_pairs(trace)
    np.testing.assert_array_equal(pair[:, :,0],trace['norm_diff'])
    np.testing.assert_array_equal(pair[0,:,1],trace['initial_previous'])
    np.testing.assert_array_equal(pair[1:,:,1],trace['norm_diff'][:-1])


def test_collector_does_not_join_different_episodes():
    class FakeEnv:
        def __init__(self):
            self.error=torch.zeros(2,172)
            self.zeros=torch.zeros_like(self.error)
            self.step_index=0
        def set_mode(self,mode):
            pass
        def reset(self,ids=None):
            if ids is None:
                self.error.zero_()
            else:
                self.error[ids]=100.
            return self.observation()
        def observation(self):
            return torch.zeros(2,1),{'disc_obs_demo':self.error,'disc_obs':self.zeros}
        def get_motion_phase(self):
            return torch.zeros(2)
        def step(self,action):
            self.step_index+=1
            self.error.add_(1.)
            obs,info=self.observation()
            done=torch.tensor([int(self.step_index==2),0])
            return obs,torch.zeros(2),done,info
    identity=SimpleNamespace(normalize=lambda x:x,unnormalize=lambda x:x)
    agent=SimpleNamespace(load=lambda path:None,eval=lambda:None,_obs_norm=identity,
        _a_norm=identity,_model=SimpleNamespace(eval_actor=lambda obs:
            SimpleNamespace(mean=torch.zeros(2,1),stddev=torch.zeros(2,1))))
    result=collect_training(FakeEnv(),agent,2999,123,3)
    pairs=result['pair'].reshape(3,2,2,172)
    torch.testing.assert_close(pairs[:,0,:,0],torch.tensor([[1.,0.],[2.,1.],[101.,100.]]))
    torch.testing.assert_close(pairs[:,1,:,0],torch.tensor([[1.,0.],[2.,1.],[3.,2.]]))
    assert result['resets']==1
