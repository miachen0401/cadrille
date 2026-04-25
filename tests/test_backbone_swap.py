"""Smoke tests for the backbone-swap factory in common.model.

Verifies that:
  - default `Cadrille` still subclasses Qwen2VLForConditionalGeneration
    (backward-compatibility for existing checkpoints + RL eval pipeline).
  - `get_cadrille_class('qwen2_5_vl')` builds a Cadrille subclass of the
    Qwen2.5-VL parent, with the same forward / compute_sequence_logprob /
    prepare_inputs_for_generation API surface.
  - unknown backbones raise ValueError.
  - Qwen3-VL raises ImportError on transformers versions that don't ship it
    (forward-compat path; passes once transformers is upgraded).

No GPU / model weights involved — class introspection only.
"""
import pytest


def test_default_cadrille_is_qwen2vl():
    from common.model import Cadrille
    from transformers import Qwen2VLForConditionalGeneration
    assert issubclass(Cadrille, Qwen2VLForConditionalGeneration)


def test_default_cadrille_has_methods():
    from common.model import Cadrille
    for m in ('forward', 'compute_sequence_logprob', 'prepare_inputs_for_generation'):
        assert hasattr(Cadrille, m), f'Cadrille missing {m!r}'


def test_get_cadrille_qwen2vl_is_cached_default():
    from common.model import Cadrille, get_cadrille_class
    # Multiple aliases all resolve to the same cached default class
    assert get_cadrille_class('qwen2_vl') is Cadrille
    assert get_cadrille_class('qwen2-vl') is Cadrille
    assert get_cadrille_class('Qwen2VL') is Cadrille


def test_get_cadrille_qwen25vl():
    from common.model import get_cadrille_class
    from transformers import Qwen2_5_VLForConditionalGeneration
    Cad25 = get_cadrille_class('qwen2_5_vl')
    assert issubclass(Cad25, Qwen2_5_VLForConditionalGeneration)
    assert hasattr(Cad25, 'forward')
    assert hasattr(Cad25, 'compute_sequence_logprob')
    # Aliases work
    assert get_cadrille_class('qwen2.5-vl').__name__ == Cad25.__name__


def test_get_cadrille_unknown_raises():
    from common.model import get_cadrille_class
    with pytest.raises(ValueError, match='unknown backbone'):
        get_cadrille_class('llava')


def test_get_cadrille_qwen3vl_forward_compat():
    """Should ImportError on transformers without Qwen3-VL, succeed once shipped."""
    from common.model import get_cadrille_class
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except ImportError:
        # Current transformers (4.50.3) — verify graceful error.
        with pytest.raises(ImportError, match='Qwen3-VL'):
            get_cadrille_class('qwen3_vl')
    else:
        # Future transformers — verify it builds.
        cls = get_cadrille_class('qwen3_vl')
        assert issubclass(cls, Qwen3VLForConditionalGeneration)
