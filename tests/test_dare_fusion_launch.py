import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from launch_cpl_fusion_climb import audit_smoke


def test_smoke_requires_checkpoint_not_just_exit_code(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_smoke(tmp_path, 20)


def test_smoke_requires_budget_optimizer_steps_and_finite_actual_model(tmp_path):
    checkpoint = dict(trainer_state=dict(next_iter=20, sample_count=20 * 8192 * 32, resume_count=0),
        optimizer_state_dicts={name: dict(steps=20 * count) for name, count in
            [('_actor_optimizer', 40), ('_critic_optimizer', 32), ('_disc_optimizer', 32)]},
        model_state_dict={'_model._disc_logits.parametrizations.weight.original': torch.ones(1, 512)})
    # Fixture is test-generated, not an edit of a repository file.
    (tmp_path / 'console.log').write_text('Completed fixture')
    torch.save(checkpoint, tmp_path / 'checkpoint.pt')
    assert audit_smoke(tmp_path, 20)['model_tensors_finite']
    checkpoint['optimizer_state_dicts']['_actor_optimizer']['steps'] = 0
    torch.save(checkpoint, tmp_path / 'checkpoint.pt')
    with pytest.raises(ValueError, match='optimizer'):
        audit_smoke(tmp_path, 20)
