#!/usr/bin/env python3
"""Continue both arms from their own 100-update Backflip checkpoints.

Copies the trained arms before advancing either one. Training resumes using
each arm's frozen source, with a matching simulator restart boundary.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import run_mira_exact_path_backflip as pair


TRAINER, ROOT = pair.trainer, pair.ROOT
START, STOP = 100, 300
SOURCE = ROOT / "output/diagnostics/mira_exact_path_pair_backflip"
ARM_DIRECTORIES = {"exact_path": "exact_path", "per_layer": "layerwise"}
AUDITED = ("checkpoint.pt", "model.pt", "source_manifest.json", "status.json",
           "initial_state.json", "first_rollout_before_update.json", "progress.json",
           "train_metrics.jsonl", "training_rounds.jsonl", "constraint_rounds.jsonl",
           "frozen_agent.yaml", "frozen_env.yaml", "frozen_engine.yaml")


def files_at(root):
    return {arm: root / ARM_DIRECTORIES[arm] for arm in pair.ARMS}


def audit(directories):
    differences = pair.verify_pair_sources(directories, START)
    hashes = {}
    for arm, directory in directories.items():
        status = pair.read_json(directory / "status.json")
        checkpoint = status["checkpoint_check"]
        assert status["status"] == "completed" and status["updates"] == START
        assert status["from_zero"] and not status["resumed"]
        assert checkpoint["status"] == "passed" and checkpoint["saved_next_iter"] == START
        assert checkpoint["saved_samples"] == START * TRAINER.NUM_ENVS * TRAINER.STEPS
        assert checkpoint["model_and_optimizer_state_reload"]
        hashes[arm] = {name: TRAINER.sha(directory / name) for name in AUDITED}
        manifest = pair.read_json(directory / "source_manifest.json")
        for kind in TRAINER.CONFIGS:
            assert manifest["inputs"]["configs"][kind]["sha256"] == hashes[arm]["frozen_" + kind + ".yaml"]
    return differences, hashes


def fork(originals, destinations, origin_hashes):
    for arm, source in originals.items():
        destination = destinations[arm]
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".training.lock", "__pycache__"))
        for name, expected in origin_hashes[arm].items():
            assert TRAINER.sha(destination / name) == expected, (arm, name)
        index = destination / "source_manifest.json"
        manifest = pair.read_json(index)
        # The frozen worker insists the actual source root matches its manifest.
        # No frozen code or config contents are modified. Preserve the prior
        # manifest's original hash in the continuation protocol.
        manifest["frozen_code_root"] = str(destination / "source_snapshot")
        TRAINER.save(index, manifest)
    return pair.verify_pair_sources(destinations, START)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Read donor manifests and artifacts; no writing or GPU")
    mode.add_argument("--stage-only", action="store_true", help="Copy and verify both runs without starting training")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out_dir.resolve()
    if source == out or source in out.parents or out in source.parents:
        parser.error("The output must be a separate directory outside the source tree")
    originals, destinations = files_at(source), files_at(out)
    differences, source_hashes = audit(originals)
    if args.preflight:
        print(f"Verified both donor checkpoints at update {START} and matching frozen training inputs.")
        return
    out.mkdir(parents=True, exist_ok=False)

    def status(phase, **data):
        TRAINER.save(out / "pair_status.json", {"status": phase, "pid": os.getpid(),
                     "starting_updates": START, "target_updates": STOP, **data})

    try:
        status("copying")
        assert fork(originals, destinations, source_hashes) == differences
        protocol = {"source_pair": str(source), "starting_updates": START, "target_updates": STOP,
                    "frozen_source_differences": differences, "resumed_own_checkpoint_per_arm": True,
                    "same_restart_boundary": True, "physics_state_restored": False,
                    "source_sha256": source_hashes}
        TRAINER.save(out / "continuation_protocol.json", protocol)
        status("staged")
        if args.stage_only:
            return
        for arm in pair.ARMS:
            destination = destinations[arm]
            status("training", arm=arm)
            pair.run(destination, (destination / "source_snapshot" / TRAINER.RUNNER_REL,
                     "--resume", "--updates", STOP, "--out-dir", destination), "continuation.console.log")
            finished = pair.read_json(destination / "status.json")
            assert (finished["status"], finished["updates"], finished["resumed"]) == ("completed", STOP, True)
            assert finished["frozen_sources_unchanged"]
            assert finished["checkpoint_check"]["saved_next_iter"] == STOP
            assert finished["samples"] == STOP * TRAINER.NUM_ENVS * TRAINER.STEPS
            restored = pair.read_json(destination / "resume_segments/001/restored_state.json")
            assert restored["model_state_exact_match"] and restored["restored_updates"] == START
            assert not restored["physics_state_restored"]
        eval_dirs = {arm: out / "evaluations" / arm for arm in pair.ARMS}
        for arm, destination in destinations.items():
            status("evaluating", arm=arm)
            eval_dir = eval_dirs[arm]
            eval_dir.mkdir(parents=True)
            pair.run(eval_dir, ("-c", pair.EVALUATE, destinations["exact_path"] / "source_snapshot", ROOT,
                     destination / "source_snapshot", "--model-file", destination / "model.pt",
                     "--env-config", destination / "frozen_env.yaml", "--agent-config", destination / "frozen_agent.yaml",
                     "--engine-config", destination / "frozen_engine.yaml", "--method", "MIRA", "--motion", "backflip",
                     "--num-envs", pair.EVAL_ENVS, "--steps", pair.EVAL_STEPS, "--start-mode", "phase0", "--seed", 0,
                     "--out-dir", eval_dir), "console.log")
        assert audit(originals) == (differences, source_hashes)
        TRAINER.save(out / "comparison.json", {**pair.compare(destinations, eval_dirs),
                     "continuation_protocol": protocol, "source_pair_unchanged": True})
        status("completed")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
