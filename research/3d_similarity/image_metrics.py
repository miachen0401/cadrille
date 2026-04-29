"""Pairwise image similarity helpers — operate on PIL Images of arbitrary size.

In our pipeline both inputs are 268×268 4-view collages produced by
`common.meshio.render_img` (this matches the BenchCAD upstream `composite_png`).
Each metric is computed on (gt_img, pred_img) and returns one float — higher
is more similar EXCEPT LPIPS (lower is more similar; we negate for sign-coherence
in tables when convenient).

All learned models lazy-load on first call and are cached on the function
object so a long for-loop pays the load cost once.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pixel / classical
# ---------------------------------------------------------------------------

def psnr(gt_img, pred_img) -> Optional[float]:
    """Pixel PSNR. Both images resized to gt_img.size if they differ."""
    try:
        import PIL.Image  # noqa: F401
        if gt_img.size != pred_img.size:
            pred_img = pred_img.resize(gt_img.size, resample=2)  # BILINEAR=2
        a = np.asarray(gt_img.convert('RGB'), dtype=np.float32) / 255.0
        b = np.asarray(pred_img.convert('RGB'), dtype=np.float32) / 255.0
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return 99.0
        return float(-10.0 * math.log10(mse))
    except Exception:
        return None


def ssim_score(gt_img, pred_img) -> Optional[float]:
    """SSIM. Returns single score in [-1, 1]; higher = more similar."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        if gt_img.size != pred_img.size:
            pred_img = pred_img.resize(gt_img.size, resample=2)
        a = np.asarray(gt_img.convert('RGB'), dtype=np.float32) / 255.0
        b = np.asarray(pred_img.convert('RGB'), dtype=np.float32) / 255.0
        # channel_axis=-1 for HxWxC
        return float(ssim_fn(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Learned: LPIPS (perceptual; lower=better)
# ---------------------------------------------------------------------------

def lpips_distance(gt_img, pred_img, net: str = 'alex') -> Optional[float]:
    """LPIPS distance with AlexNet features (light, ~5 MB). Lower = more similar."""
    try:
        import torch
        cache = getattr(lpips_distance, '_cache', None)
        if cache is None:
            import lpips
            model = lpips.LPIPS(net=net, verbose=False)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device).eval()
            lpips_distance._cache = (model, device)
        model, device = lpips_distance._cache
        if gt_img.size != pred_img.size:
            pred_img = pred_img.resize(gt_img.size, resample=2)
        def _to_tensor(img):
            arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
            arr = arr * 2 - 1   # LPIPS expects [-1, 1]
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            return t
        with torch.no_grad():
            d = model(_to_tensor(gt_img), _to_tensor(pred_img))
        return float(d.item())
    except Exception as e:
        print(f'  lpips err: {e}', flush=True)
        return None


# ---------------------------------------------------------------------------
# Learned: DINOv2 / CLIP  (cosine of [CLS] embeddings; higher=better)
# ---------------------------------------------------------------------------

def _vlm_cos(gt_img, pred_img, model_id: str, cache_attr: str) -> Optional[float]:
    """Generic ViT-encoder cosine helper. Caches on the calling fn."""
    try:
        import torch
        # The cache is keyed by attr name on the *outer* fn — fetch via globals
        cache = globals().get(cache_attr)
        if cache is None:
            from transformers import AutoModel, AutoImageProcessor
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id).to(device).eval()
            cache = (model, processor, device)
            globals()[cache_attr] = cache
        model, processor, device = cache
        with torch.no_grad():
            inp = processor(images=[gt_img.convert('RGB'), pred_img.convert('RGB')],
                            return_tensors='pt').to(device)
            out = model(**inp)
            emb = out.pooler_output if getattr(out, 'pooler_output', None) is not None \
                  else out.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            sim = float((emb[0] * emb[1]).sum())
        return sim
    except Exception as e:
        print(f'  {model_id} err: {e}', flush=True)
        return None


def dino_cos(gt_img, pred_img,
             model_id: str = 'facebook/dinov2-small') -> Optional[float]:
    """DINOv2 [CLS] cosine. ~22M params, ~50ms / pair on GPU."""
    return _vlm_cos(gt_img, pred_img, model_id, '_DINO_CACHE')


def clip_cos(gt_img, pred_img,
             model_id: str = 'openai/clip-vit-base-patch32') -> Optional[float]:
    """CLIP image-image cosine. Vision encoder only (ignores text)."""
    try:
        import torch
        cache = globals().get('_CLIP_CACHE')
        if cache is None:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            processor = CLIPImageProcessor.from_pretrained(model_id)
            model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
            cache = (model, processor, device)
            globals()['_CLIP_CACHE'] = cache
        model, processor, device = cache
        with torch.no_grad():
            inp = processor(images=[gt_img.convert('RGB'), pred_img.convert('RGB')],
                            return_tensors='pt').to(device)
            out = model(**inp)
            emb = out.image_embeds
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return float((emb[0] * emb[1]).sum())
    except Exception as e:
        print(f'  clip err: {e}', flush=True)
        return None
