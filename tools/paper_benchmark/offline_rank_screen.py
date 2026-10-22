#!/usr/bin/env python3
"""Offline BCE versus BCE+physics-ranking screen for a frozen DARE model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.util as util  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--env-config", default="data/envs/paper_benchmark/dare_climb_env.yaml")
    p.add_argument("--agent-config", default="data/agents/dare_10layer_cpl_climb_agent.yaml")
    p.add_argument("--engine-config", default="data/engines/isaac_lab_engine.yaml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--updates", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-file", required=True)
    return p.parse_args()


def load_samples(rollout_dir, normalizer, device):
    root = Path(rollout_dir)
    ts = np.load(root / "timeseries.npz")
    ep = np.load(root / "episodes.npz")
    raw = ts["disc_diff"].astype(np.float32)
    alive = ts["alive"].astype(bool)
    start = ep["start_phase"].astype(np.float32)
    labels = ep["completion"].astype(np.int64)
    dt = float(ts["time_seconds"][0])
    motion_length = 9.9666666667
    phase = np.clip(start[None, :] + dt * np.arange(raw.shape[0])[:, None] / motion_length, 0, 1)
    valid = alive & np.isfinite(raw).all(-1)
    idx_t, idx_e = np.where(valid)
    raw_t = torch.from_numpy(raw[idx_t, idx_e]).to(device)
    norm = normalizer.normalize(raw_t).detach()
    phase_t = torch.from_numpy(phase[idx_t, idx_e]).to(device)
    label_t = torch.from_numpy(labels[idx_e]).to(device=device, dtype=torch.long)
    episode_t = torch.from_numpy(idx_e).to(device=device, dtype=torch.long)
    return norm, phase_t, label_t, episode_t


def make_pairs(norm, phase, labels, episodes, seed):
    rng = np.random.default_rng(seed)
    magnitude = torch.linalg.vector_norm(norm, dim=-1).detach().cpu().numpy()
    phase_np = phase.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    episode_np = episodes.detach().cpu().numpy()
    unique = np.unique(episode_np)
    rng.shuffle(unique)
    split = max(1, int(0.8 * unique.size))
    train_eps = set(unique[:split].tolist())
    test_eps = set(unique[split:].tolist())
    train_mask = np.asarray([x in train_eps for x in episode_np])
    test_mask = np.asarray([x in test_eps for x in episode_np])

    pairs = {"train": [], "test": []}
    for name, mask in (("train", train_mask), ("test", test_mask)):
        for phase_bin in range(64):
            ids = np.flatnonzero(mask & (np.floor(phase_np * 64).astype(int) == phase_bin))
            pos = ids[labels_np[ids] == 1]
            neg = ids[labels_np[ids] == 0]
            if not len(pos) or not len(neg):
                continue
            for i in pos:
                close = neg[np.abs(magnitude[neg] - magnitude[i]) <= 0.05]
                if len(close):
                    j = int(rng.choice(close))
                    pairs[name].append((int(i), j))
    if not pairs["train"] or not pairs["test"]:
        raise RuntimeError("insufficient cross-outcome pairs for grouped screen")
    return pairs, train_mask, test_mask


def evaluate(model, norm, pairs, split, batch_size, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    pair = np.asarray(pairs[split], dtype=np.int64)
    with torch.no_grad():
        logits = model.eval_disc_raw(norm).squeeze(-1)
        # Anchor logit is evaluated separately to avoid spectral-normalization
        # state changes during the summary pass.
        anchor = model.eval_disc_raw(torch.zeros_like(norm[:1])).squeeze(-1)
        cls = 0.5 * (torch.nn.functional.softplus(-anchor).mean() + torch.nn.functional.softplus(logits).mean())
        take = min(pair.shape[0], 200000)
        chosen = pair[rng.choice(pair.shape[0], take, replace=False)]
        pos = logits[torch.from_numpy(chosen[:, 0]).to(norm.device)]
        neg = logits[torch.from_numpy(chosen[:, 1]).to(norm.device)]
        rank_loss = torch.nn.functional.softplus(neg - pos).mean()
        accuracy = (pos > neg).float().mean()
    return {"classification_loss": float(cls), "rank_loss": float(rank_loss),
            "pair_accuracy": float(accuracy), "pairs": int(take),
            "anchor_logit": float(anchor.mean())}


def train_rank(model, norm, pairs, updates, batch_size, seed, anchor_target):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=2.5e-4, momentum=0.9, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    pair = np.asarray(pairs["train"], dtype=np.int64)
    anchor = torch.zeros((1, norm.shape[-1]), device=norm.device)
    for _ in range(updates):
        chosen = pair[rng.integers(pair.shape[0], size=batch_size)]
        pos_idx = torch.from_numpy(chosen[:, 0]).to(norm.device)
        neg_idx = torch.from_numpy(chosen[:, 1]).to(norm.device)
        pos = model.eval_disc_raw(anchor).squeeze(-1)
        pos_batch = model.eval_disc_raw(norm[pos_idx]).squeeze(-1)
        neg_batch = model.eval_disc_raw(norm[neg_idx]).squeeze(-1)
        cls = 0.5 * (torch.nn.functional.softplus(-pos).mean() + torch.nn.functional.softplus(neg_batch).mean())
        rank = torch.nn.functional.softplus(neg_batch - pos_batch).mean()
        loss = cls + rank
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        # The zero-differential anchor is part of ADD's reward calibration.
        # Pairwise loss is invariant to a global logit shift, so preserve the
        # original anchor exactly instead of introducing a new scale weight.
        with torch.no_grad():
            current_anchor = model.eval_disc_raw(anchor).mean()
            model._disc_logits.bias.add_(anchor_target - current_anchor)
    return model


def main():
    a = parse_args()
    out = Path(a.out_file)
    if out.exists():
        raise FileExistsError(out)
    util.set_rand_seed(a.seed)
    mp_util.init(0, 1, a.device, 6393)
    env = env_builder.build_env(a.env_config, a.engine_config, 1, a.device, False, False)
    agent = agent_builder.build_agent(a.agent_config, env, a.device)
    agent.load(a.model_file)
    agent.eval()
    normalizer = agent._disc_obs_norm
    norm, phase, labels, episodes = load_samples(a.rollout_dir, normalizer, a.device)
    pairs, train_mask, test_mask = make_pairs(norm, phase, labels, episodes, a.seed)
    baseline = copy.deepcopy(agent._model).to(a.device)
    ranked = copy.deepcopy(agent._model).to(a.device)
    with torch.no_grad():
        anchor_target = baseline.eval_disc_raw(
            torch.zeros((1, norm.shape[-1]), device=a.device)).mean()
    ranked = train_rank(ranked, norm,
                        {"train": pairs["train"], "test": pairs["test"]},
                        a.updates, a.batch_size, a.seed, anchor_target)
    # The pair indices above are local only for the train tensor; rebuild test
    # evaluation on the full tensor and report the held-out pair list directly.
    baseline_eval = evaluate(baseline, norm, pairs, "test", a.batch_size, a.seed)
    ranked_eval = evaluate(ranked, norm, pairs, "test", a.batch_size, a.seed)
    report = {
        "samples": int(norm.shape[0]),
        "episodes": int(torch.unique(episodes).numel()),
        "train_pairs": len(pairs["train"]),
        "test_pairs": len(pairs["test"]),
        "updates": a.updates,
        "loss_definition": "BCE + 1.0 * softplus(logit_negative - logit_success)",
        "baseline_test": baseline_eval,
        "ranked_test": ranked_eval,
        "pass": (ranked_eval["pair_accuracy"] > baseline_eval["pair_accuracy"]
                 and ranked_eval["classification_loss"] <= baseline_eval["classification_loss"] * 1.05),
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
