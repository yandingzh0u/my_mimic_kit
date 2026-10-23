import torch

from learning.phase_stratified_sampler import PhaseStratifiedSampler


def test_equal_quota_and_phase_mean_weights():
    phase = torch.tensor([[0.01, 0.26, 0.51, 0.76],
                          [0.03, 0.31, 0.55, 0.79]])
    sampler = PhaseStratifiedSampler(phase, num_bins=4)
    indices, weights = sampler.sample(40)
    bins = (phase.flatten() * 4).long()[indices]
    assert sorted(torch.bincount(bins, minlength=4).tolist()) == [10, 10, 10, 10]
    assert torch.allclose(weights[bins == 0], torch.full((10,), 0.1))
    assert torch.allclose(weights.sum(), torch.tensor(4.0))


def test_eligible_mask_excludes_deterministic_actions():
    phase = torch.tensor([[0.01, 0.26, 0.51, 0.76],
                          [0.03, 0.31, 0.55, 0.79]])
    eligible = torch.tensor([True, False, True, True, True, True, False, True])
    sampler = PhaseStratifiedSampler(phase, num_bins=4)
    indices, _ = sampler.sample(32, eligible)
    assert torch.all(eligible[indices])


def test_sample_indices_preserves_explicit_order(tmp_path):
    from learning.experience_buffer import ExperienceBuffer

    buffer = ExperienceBuffer(2, 3, "cpu")
    buffer.record("x", torch.tensor([[0.0], [1.0], [2.0]]))
    buffer.inc()
    buffer.record("x", torch.tensor([[3.0], [4.0], [5.0]]))
    buffer.inc()
    batch = buffer.sample_indices(torch.tensor([5, 0, 3]))
    assert batch["x"].flatten().tolist() == [5.0, 0.0, 3.0]
