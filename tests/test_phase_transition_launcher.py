from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_phase_transition_formal_launcher_syntax_and_contract():
    script = ROOT / "tools/phase_transition_critic/run_roll_2k.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in text
    assert "flock -n 9" in text
    assert "flock -n 8" in text
    assert "git status --porcelain --untracked-files=normal" in text
    assert 'git_branch="$(git symbolic-ref' in text
    assert 'git_commit="$(git rev-parse --verify HEAD)"' in text
    assert "manifest.env" in text
    assert "MANIFEST_ARG_SHA" in text
    assert "MANIFEST_AGENT_SHA" in text
    assert "MANIFEST_ENV_SHA" in text
    assert "MANIFEST_ENGINE_SHA" in text
    assert "MANIFEST_PYTHON_PATH" in text
    assert "MANIFEST_RUNTIME" in text
    assert '"gpu0_name": props.name' in text

    assert "num_envs=8192" in text
    assert "steps_per_iter=32" in text
    assert "target_iters=2000" in text
    assert "target_samples=$((num_envs * steps_per_iter * target_iters))" in text
    assert "--resume_file" in text
    assert "checkpoint_samples" in text
    assert 'metadata.get("agent_class") != "PhaseTransitionCriticAgent"' in text
    assert '"_actor_optimizer", "_critic_optimizer", "_disc_optimizer"' in text
    assert 'if set(replays) != {"_disc_buffer"}' in text
    for field in (
        "sim_state", "sim_motion", "ref_state", "ref_motion",
        "motion_id", "motion_phase", "motion_is_wrap",
    ):
        assert f'"{field}"' in text

    assert '>> "$out_dir/console.log" 2>&1 &' in text
    assert 'write_atomic "$out_dir/launcher.pid"' in text
    assert 'write_atomic "$out_dir/train.pid"' in text
    assert 'write_atomic "$out_dir/FAILED"' in text
    assert 'write_atomic "$out_dir/DONE"' in text
    assert 'write_atomic "$out_dir/STATUS"' in text

    assert "tools/paper_eval/evaluate_checkpoint.py" in text
    assert "--num-envs 256" in text
    assert "--steps 300" in text
    assert "--start-mode phase0" in text
    assert "--condition nominal" in text
    assert 'eval_dir="$out_dir/eval/final"' in text
    assert "summary.json" in text
    assert "episodes.npz" in text
    assert "timeseries.npz" in text


def test_phase_transition_formal_args_match_launcher_budget():
    args = (
        ROOT / "args/phase_transition_critic_humanoid_roll_2k_8192_args.txt"
    ).read_text(encoding="utf-8")
    assert "--num_envs 8192" in args
    assert "--max_samples 524288000" in args
    assert "--out_dir output/phase_transition_critic_roll_2k_8192_seed0" in args
    assert "--rand_seed 0" in args
    assert "--save_int_models true" in args
