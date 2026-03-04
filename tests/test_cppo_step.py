"""Unit test for cppo_step with a synthetic point-cloud item.

Stubs out compute_rewards_parallel so no mesh files or CadQuery subprocess
are needed. Verifies the full forward+backward pass runs without error and
returns the expected metric keys.

Run:
    pytest tests/test_cppo_step.py -v
"""

import os
import sys
import copy
import types

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SFT_CHECKPOINT = './checkpoints/cadrille-sft'


def _stub_rewards(codes, mesh_paths, workers=1):
    """Always return -10 (all invalid) — avoids real CadQuery subprocess."""
    return [-10.0] * len(codes)


def _make_args(G=2, max_new_tokens=16):
    a = types.SimpleNamespace()
    a.G = G
    a.top_N = 1
    a.eps_high = 0.1
    a.eps_low = 0.1
    a.batch_updates = 1
    a.max_new_tokens = max_new_tokens
    a.reward_workers = 1
    a.sequential_generation = True
    return a


@pytest.fixture(scope='module')
def model_and_processor():
    """Load model + processor once per test module (expensive)."""
    if not os.path.isdir(SFT_CHECKPOINT):
        pytest.skip(f'SFT checkpoint not found at {SFT_CHECKPOINT}')

    from transformers import AutoProcessor
    from cadrille import Cadrille

    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    model = Cadrille.from_pretrained(
        SFT_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto')

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=3e-5)
    except ImportError:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    old_model = copy.deepcopy(model).cpu()
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    return model, old_model, optimizer, processor


def _synthetic_item():
    """Minimal point-cloud item (no real mesh file needed with stub rewards)."""
    pc = np.random.randn(256, 3).astype(np.float32)
    return {
        'point_cloud': pc,
        'description': 'Generate cadquery code',
        'file_name': 'test_cube',
        'gt_mesh_path': '/nonexistent/test.stl',
    }


def test_cppo_step_runs(model_and_processor):
    """cppo_step completes without error and returns all expected metrics."""
    import rl.algorithms.cppo as cppo_mod

    model, old_model, optimizer, processor = model_and_processor
    orig_fn = cppo_mod.compute_rewards_parallel
    cppo_mod.compute_rewards_parallel = _stub_rewards

    try:
        metrics = cppo_mod.cppo_step(
            model, old_model, optimizer, _synthetic_item(), processor, _make_args())
    finally:
        cppo_mod.compute_rewards_parallel = orig_fn

    expected_keys = [
        'train/loss', 'train/mean_reward', 'train/reward_std', 'train/entropy',
        'train/clip_fraction', 'train/ratio_mean', 'train/gen_seconds',
        'train/q_pp', 'train/q_pn', 'train/q_np', 'train/q_nn',
    ]
    for k in expected_keys:
        assert k in metrics, f'Missing key: {k}'

    assert isinstance(metrics['train/gen_seconds'], float)
    assert metrics['train/mean_reward'] == pytest.approx(-10.0)


def test_cppo_step_degenerate_rewards(model_and_processor):
    """When all rewards equal, adv_pos_frac and q_* are NaN (not 0)."""
    import rl.algorithms.cppo as cppo_mod

    model, old_model, optimizer, processor = model_and_processor
    orig_fn = cppo_mod.compute_rewards_parallel
    cppo_mod.compute_rewards_parallel = _stub_rewards

    try:
        metrics = cppo_mod.cppo_step(
            model, old_model, optimizer, _synthetic_item(), processor, _make_args())
    finally:
        cppo_mod.compute_rewards_parallel = orig_fn

    import math
    assert math.isnan(metrics['train/adv_pos_frac']), 'Expected NaN for degenerate step'
    assert math.isnan(metrics['train/q_pp']), 'Expected NaN for degenerate step'
