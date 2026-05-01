"""Distribution-level metrics — FID, KID, CLIP R-Precision.

These are NOT per-case. They take a *set* of predicted images and compare
against a *set* of real images to score "how realistic / on-distribution
is this model's output". Intended use:
  - one number per model
  - drop into the headline baselines table

Implementations follow the standard formulations and avoid third-party
packages (no clean-fid / pytorch-fid) so the surface stays small.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Inception-V3 pool features (2048-d) — used by FID and KID
# ---------------------------------------------------------------------------

def inception_pool_features(images, device: str = 'cuda', batch_size: int = 32):
    """Run Inception-V3 (ImageNet-pretrained) on a list of PIL Images and
    return a (N, 2048) numpy array of pool3 features.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision import transforms

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True).to(device).eval()
    # Override fc with identity to grab 2048-d pool features
    model.fc = torch.nn.Identity()

    tx = transforms.Compose([
        transforms.Resize(299, antialias=True),
        transforms.CenterCrop(299),
        transforms.ToTensor(),                              # [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    n = len(images)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = [tx(im.convert('RGB')) for im in images[i:i + batch_size]]
            x = torch.stack(batch).to(device)
            y = model(x)
            feats.append(y.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# CLIP image features — used by CLIP R-Precision
# ---------------------------------------------------------------------------

def clip_image_features(images, device: str = 'cuda', batch_size: int = 32,
                        model_id: str = 'openai/clip-vit-base-patch32'):
    """Run CLIP vision encoder on a list of PIL Images, return (N, dim) numpy.
    Uses safetensors weights so it works on torch < 2.6 (CVE-2025-32434).
    """
    import torch
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(model_id)
    try:
        model = CLIPVisionModelWithProjection.from_pretrained(
            model_id, use_safetensors=True).to(device).eval()
    except Exception:
        # Some snapshots ship .bin only; fall back and let user upgrade torch.
        model = CLIPVisionModelWithProjection.from_pretrained(
            model_id).to(device).eval()

    feats = []
    n = len(images)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = [im.convert('RGB') for im in images[i:i + batch_size]]
            inp = processor(images=batch, return_tensors='pt').to(device)
            out = model(**inp)
            emb = out.image_embeds
            emb = torch.nn.functional.normalize(emb, dim=-1)
            feats.append(emb.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def fid_from_features(real_feats: np.ndarray,
                      fake_feats: np.ndarray,
                      eps: float = 1e-6) -> float:
    """Frechet Inception Distance from pre-computed feature arrays.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 sqrt(Sigma_r Sigma_f))
    """
    from scipy import linalg
    mu_r = real_feats.mean(axis=0); mu_f = fake_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_f = np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma_r + offset) @ (sigma_f + offset), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f'  WARN: imaginary FID component magnitude {np.abs(covmean.imag).max()}')
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


# ---------------------------------------------------------------------------
# KID  (squared MMD with polynomial kernel; smaller bias than FID at low N)
# ---------------------------------------------------------------------------

def kid_from_features(real_feats: np.ndarray,
                      fake_feats: np.ndarray,
                      n_subsets: int = 100,
                      subset_size: Optional[int] = None,
                      degree: int = 3,
                      gamma: Optional[float] = None,
                      coef0: float = 1.0,
                      seed: int = 42) -> Tuple[float, float]:
    """KID via polynomial kernel MMD^2 (Bińkowski et al. ICLR 2018).

    Returns (mean, std) over n_subsets random subsamples of size subset_size.
    """
    rng = np.random.RandomState(seed)
    n_r, d = real_feats.shape
    n_f    = fake_feats.shape[0]
    if subset_size is None:
        subset_size = min(1000, n_r, n_f)
    if gamma is None:
        gamma = 1.0 / d
    mmds = []
    for _ in range(n_subsets):
        r_idx = rng.choice(n_r, subset_size, replace=False)
        f_idx = rng.choice(n_f, subset_size, replace=False)
        r = real_feats[r_idx]; f = fake_feats[f_idx]
        # k(x,y) = (gamma <x,y> + coef0)^degree
        k_rr = (gamma * (r @ r.T) + coef0) ** degree
        k_ff = (gamma * (f @ f.T) + coef0) ** degree
        k_rf = (gamma * (r @ f.T) + coef0) ** degree
        m = subset_size
        # unbiased MMD^2
        mmd2 = ((k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
              + (k_ff.sum() - np.trace(k_ff)) / (m * (m - 1))
              - 2 * k_rf.mean())
        mmds.append(mmd2)
    return float(np.mean(mmds)), float(np.std(mmds))


# ---------------------------------------------------------------------------
# CLIP R-Precision  (each fake_i retrieves from {real_j}, check rank of i)
# ---------------------------------------------------------------------------

def clip_r_precision(fake_feats: np.ndarray,
                     real_feats: np.ndarray,
                     paired: bool = True,
                     ks=(1, 5, 10)) -> dict:
    """For each fake_i, retrieve top-K nearest real_j by cosine. If paired,
    fake_i corresponds to real_i — check whether i is in fake_i's top-K.

    Both inputs should be L2-normalised already (clip_image_features does this).
    Returns {'r_at_1': float, 'r_at_5': float, ...}.
    """
    n = fake_feats.shape[0]
    if paired and real_feats.shape[0] != n:
        raise ValueError(f'paired r-precision needs |fake|=|real|; got {n} vs {real_feats.shape[0]}')
    sim = fake_feats @ real_feats.T   # (n_fake, n_real)
    # for each row, sort indices by descending similarity
    rankings = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        if paired:
            top_k = rankings[:, :k]
            hits = sum(1 for i in range(n) if i in top_k[i])
            out[f'r_at_{k}'] = hits / n
        else:
            # If unpaired, no notion of correct retrieval — skip
            out[f'r_at_{k}'] = None
    return out
