"""Same-rollout target/gradient audit of fixed and control-resolution strata."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'mimickit'))
from learning.phase_sampler import PhaseSampler
from learning.phase_resolution import supported_resolution
from adaptive_phase_live import gradient

BASE = ROOT/'output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012'
NEW = ROOT/'output/paper_benchmark/dare_10layer_cpl_twostage_climb_3000_8192_seed0_headless_20261025'


def full_gradient(agent, data, weights, adv):
    idx = (weights > 0).nonzero().flatten()
    return gradient(agent, data, weights[idx], adv, idx, chunk=8192).double()


def compare(a, b):
    return dict(relative_gradient_change=float((a-b).norm()/b.norm().clamp_min(1e-20)),
                cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20)))


def audit(agent):
    p = agent._exp_buffer.get_data_flat('diag_phase').clone()
    mask = agent._exp_buffer.get_data_flat('rand_action_mask') == 1
    data = {key: agent._exp_buffer.get_data_flat(key).detach()
            for key in ('obs','action','a_logp','rand_action_mask')}
    n, ba, bc, f = len(p), 4*agent.get_num_envs(), 2*agent.get_num_envs(), agent._phase_control_cells
    k = supported_resolution(p, f, [ba,bc], [mask,None])
    with torch.no_grad():
        raw = agent._exp_buffer.get_data_flat('tar_val') - agent._model.eval_critic(
            agent._obs_norm.normalize(data['obs'])).squeeze(-1)
    targets = {}
    samplers = {}
    for name, cells in [('fixed8',8),('native299',f),('supported',k)]:
        sampler = PhaseSampler(p, cells, bc, mask)
        samplers[name] = sampler
        weights = sampler.full_weights()
        mean = (weights*raw).sum()
        std = ((weights*raw.square()).sum()-mean.square()).clamp_min(0).sqrt()
        adv = ((raw-mean)/std.clamp_min(1e-5)).clamp(-agent._norm_adv_clip,agent._norm_adv_clip).float()
        targets[name] = (weights,adv)
    result = dict(population=n,actor_batch=ba,critic_batch=bc,supported_bins=k,schemes={})
    gs = {}
    w8, a8 = targets['fixed8']
    for name,(weights,adv) in targets.items():
        sampler = samplers[name]
        counts = sampler.counts
        quota = (min(ba,len(sampler.indices))+sampler.num_present-1)//sampler.num_present
        rare = counts < quota
        rare_mask = rare[sampler.labels]
        rare_indices = sampler.indices[rare_mask]
        gs[name] = full_gradient(agent,data,weights,adv)
        result['schemes'][name] = dict(bins=sampler.num_bins,minimum_pool=int(counts.min()),
            maximum_pool=int(counts.max()),ceil_actor_quota=quota,
            underfilled_bins=int(rare.sum()),underfilled_population=len(rare_indices),
            underfilled_population_fraction=len(rare_indices)/n,
            underfilled_target_mass=float(weights[rare_indices].sum()),
            population_ess=float(1/weights.square().sum()),maximum_relative_population_weight=float(n*weights.max()),
            advantage_mean=float((weights*raw).sum()),advantage_std=float(((weights*raw.square()).sum()-(weights*raw).sum().square()).sqrt()),
            full_actor_gradient_norm=float(gs[name].norm()),
            bin_counts=counts.tolist(),first_ten_cell_counts=counts[:10].tolist())
        if name!='fixed8':
            result['schemes'][name]['vs_fixed8'] = compare(gs[name],gs['fixed8'])
        if name=='native299':
            g_fixed_stats = full_gradient(agent,data,weights,a8)
            result['fine_weights_with_fixed8_advantage_stats'] = compare(g_fixed_stats,gs['fixed8'])
            if len(rare_indices):
                gr = gradient(agent,data,weights[rare_indices],adv,rare_indices,chunk=8192).double()*weights[rare_indices].sum()
                result['fine_underfilled_gradient'] = dict(norm=float(gr.norm()),
                    relative_norm=float(gr.norm()/gs[name].norm()),
                    cosine_with_full=float(torch.dot(gr,gs[name])/(gr.norm()*gs[name].norm()).clamp_min(1e-20)))
    _,_,counts=samplers['native299'].sample(ba)
    native = samplers['native299']
    result['native_single_batch'] = dict(unique_expected=min(ba,n),minimum_per_draw_weight=float((native.target_mass/counts).min()),maximum_per_draw_weight=float((native.target_mass/counts).max()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=8192)
    parser.add_argument('--checkpoints',nargs='+',default=['baseline:2999','twostage:2000','twostage:2999'])
    parser.add_argument('--reset-modes',nargs='+',default=['iid','coverage'])
    args=parser.parse_args()
    os.chdir(ROOT)
    args.out.mkdir(parents=True,exist_ok=True)
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util
    torch.set_num_threads(4)
    mp_util.init(0,1,'cuda:0',6413)
    util.set_rand_seed(0)
    env=env_builder.build_env(str(BASE/'env_config.yaml'),str(BASE/'engine_config.yaml'),args.num_envs,'cuda:0',visualize=False)
    agent=agent_builder.build_agent(str(NEW/'agent_config.yaml'),env,'cuda:0')
    report=dict(scope='Fresh real rollouts, frozen models; same-rollout objective and gradient comparisons, not causal retraining',num_envs=args.num_envs,cases={})
    try:
        for spec in args.checkpoints:
            source,it=spec.split(':');directory=BASE if source=='baseline' else NEW
            for reset in args.reset_modes:
                util.set_rand_seed(41025)
                agent.load(str(directory/f'int_models/model_{int(it):010d}.pt'))
                agent._init_train();agent._sample_count=10**12
                agent.eval();agent.set_mode(AgentMode.TRAIN)
                if reset=='iid':
                    if hasattr(env,'_phase_coverage'): del env._phase_coverage
                else: env.enable_phase_coverage_resets(agent._phase_control_cells)
                agent._curr_obs,agent._curr_info=env.reset()
                with torch.no_grad():
                    agent._rollout_train(agent._phase_control_cells)
                    agent._init_iter();agent._rollout_train(32);agent._build_train_data()
                row=audit(agent)
                report['cases'][spec+':'+reset]=row
                (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
                print('AUDIT '+json.dumps(dict(case=spec+':'+reset,supported_bins=row['supported_bins'],schemes={k:{x:v for x,v in d.items() if x not in ['bin_counts','first_ten_cell_counts']} for k,d in row['schemes'].items()})),flush=True)
        print('AUDIT_COMPLETE',flush=True)
    finally:
        failed=sys.exc_info()[0] is not None
        if failed: traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush();os._exit(1 if failed else 0)


if __name__=='__main__': main()
