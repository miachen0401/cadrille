"""Smoke tests for CurriculumWeightedSampler + StepTracker + step callback."""
import pytest
import torch


def test_step_tracker_basic():
    from train.sft.train import _StepTracker
    t = _StepTracker(0)
    assert t.step == 0
    t.step = 1234
    assert t.step == 1234


def test_curriculum_sampler_phase_switches():
    from train.sft.train import CurriculumWeightedSampler, _StepTracker
    # 3 phases over 6 items; each phase prefers a different range
    phases = [
        (0,    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # always pick item 0
        (1000, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # always item 1
        (2000, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # always item 2
    ]
    tracker = _StepTracker(0)
    sampler = CurriculumWeightedSampler(
        phases=phases, num_samples=8,
        lengths=[1] * 6, batch_size=2,
        step_tracker=tracker,
    )
    # Phase 0 (step=0): all draws are index 0
    tracker.step = 0
    assert all(i == 0 for i in sampler)
    # Phase 1 (step=1500): all draws are index 1
    tracker.step = 1500
    assert all(i == 1 for i in sampler)
    # Phase 2 (step=2500): all draws are index 2
    tracker.step = 2500
    assert all(i == 2 for i in sampler)
    # Below first threshold of a higher phase still uses earlier phase
    tracker.step = 999
    assert all(i == 0 for i in sampler)
    # Past the last phase remains on the last phase
    tracker.step = 999999
    assert all(i == 2 for i in sampler)


def test_curriculum_sampler_first_phase_must_start_at_zero():
    from train.sft.train import CurriculumWeightedSampler
    with pytest.raises(ValueError, match='first phase must start at step 0'):
        CurriculumWeightedSampler(
            phases=[(100, [1.0])], num_samples=1,
            lengths=[1], batch_size=1,
        )


def test_curriculum_sampler_empty_phases_rejected():
    from train.sft.train import CurriculumWeightedSampler
    with pytest.raises(ValueError, match='at least one phase'):
        CurriculumWeightedSampler(phases=[], num_samples=1, lengths=[1], batch_size=1)


def test_expand_mix_to_sample_weights_basic():
    """Sources insertion order is preserved; per-source mass = configured weight."""
    from train.sft.train import _expand_mix_to_sample_weights
    sources = {
        'benchcad':  list(range(4)),    # 4 items
        'recode20k': list(range(2)),    # 2 items
        'text2cad':  list(range(2)),    # 2 items
    }
    mix = {'benchcad': 4, 'recode20k': 1, 'text2cad': 1}
    w = _expand_mix_to_sample_weights(mix, sources)
    assert len(w) == 8
    # Each benchcad item has weight 4/4 = 1.0
    assert w[0:4] == [1.0, 1.0, 1.0, 1.0]
    # Each recode20k item has 1/2 = 0.5
    assert w[4:6] == [0.5, 0.5]
    # Each text2cad item has 1/2 = 0.5
    assert w[6:8] == [0.5, 0.5]
    # Total mass per source equals its mix weight
    assert sum(w[0:4]) == 4
    assert sum(w[4:6]) == 1
    assert sum(w[6:8]) == 1


def test_expand_mix_to_sample_weights_zero_weight_drops():
    from train.sft.train import _expand_mix_to_sample_weights
    sources = {'a': [0, 1], 'b': [0, 1]}
    mix = {'a': 0, 'b': 1}
    w = _expand_mix_to_sample_weights(mix, sources)
    assert w == [0.0, 0.0, 0.5, 0.5]
