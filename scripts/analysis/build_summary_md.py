"""Compile a single markdown report from all available cad-eval results.

Reads:
  - eval_outputs/cad_bench_722/{summary,summary_iou_24,distribution_metrics}.json
  - eval_outputs/cad_bench_722/<model>/metadata.jsonl  (per-difficulty rollup)
  - eval_outputs/cad_bench_722/iou_vs_iou24/report.md  (rotation-rescue tables)
  - eval_outputs/{deepcad,fusion360}_n300/<model>/metadata.jsonl
                                              (out-of-distribution evals)

Writes:
  - docs/cad_bench_722_baselines.md          (overwrite — full ledger)
  - eval_outputs/cad_bench_722/RESULTS.md    (snapshot for the eval-output dir)

Usage:
    uv run python scripts/analysis/build_summary_md.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'

MODEL_LABEL = {
    'cadrille_rl':         'Cadrille-rl (filapro)',
    'cadevolve_rl1':       'CADEvolve-rl1 (kulibinai)',
    'qwen25vl_3b_zs':      'Qwen2.5-VL-3B (zero-shot)',
    'cadrille_qwen3vl_v3': 'Cadrille-Q3VL-v3 (50k clean)',
}
MODEL_ORDER = ['cadrille_rl', 'cadevolve_rl1', 'qwen25vl_3b_zs',
               'cadrille_qwen3vl_v3']


def _summarize_metadata(path: Path) -> dict:
    if not path.exists(): return {}
    rs = []
    for line in open(path):
        try: rs.append(json.loads(line))
        except: pass
    if not rs: return {}
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    out = {
        'n':         len(rs),
        'n_success': len(ok),
        'exec_rate': len(ok) / len(rs) if rs else 0,
        'mean_iou':  sum(ious) / len(ious) if ious else None,
        'mean_cd':   sum(cds)  / len(cds)  if cds  else None,
    }
    # per-difficulty
    by_diff = defaultdict(list)
    for r in rs: by_diff[r.get('difficulty', '?')].append(r)
    diff = {}
    for d, grp in by_diff.items():
        grp_ok = [r for r in grp if r.get('error_type') == 'success']
        grp_iou = [r['iou'] for r in grp_ok if r.get('iou') is not None]
        diff[d] = {
            'n':         len(grp),
            'exec_rate': len(grp_ok) / len(grp) if grp else 0,
            'mean_iou':  sum(grp_iou) / len(grp_iou) if grp_iou else None,
        }
    out['by_difficulty'] = diff
    return out


def _summarize_metadata_24(path: Path) -> dict:
    """Same as _summarize_metadata but also reads iou_24 + rot_idx."""
    if not path.exists(): return {}
    rs = []
    for line in open(path):
        try: rs.append(json.loads(line))
        except: pass
    if not rs: return {}
    paired = [r for r in rs
              if r.get('iou') is not None and r.get('iou_24') is not None]
    iou1 = [r['iou']    for r in paired]
    iou24 = [r['iou_24'] for r in paired]
    rots = [r.get('rot_idx', -1) for r in paired]
    return {
        'n_paired':    len(paired),
        'mean_iou':    sum(iou1)/len(iou1) if iou1 else None,
        'mean_iou_24': sum(iou24)/len(iou24) if iou24 else None,
        'mean_delta':  sum(b - a for a, b in zip(iou1, iou24))/len(paired) if paired else None,
        'pct_rot_win': sum(1 for r in rots if r > 0)/len(rots) if rots else None,
    }


def _fmt_pct(v): return f'{v*100:.1f}%' if v is not None else '—'
def _fmt_iou(v): return f'{v:.4f}' if v is not None else '—'
def _fmt_cd(v):  return f'{v:.6f}' if v is not None else '—'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(REPO / 'docs' / 'cad_bench_722_baselines.md'))
    ap.add_argument('--snapshot', default=str(EVAL_ROOT / 'RESULTS.md'))
    args = ap.parse_args()

    lines = ['# `cad_bench_722` — multi-baseline evaluation',
             '',
             '**Dataset:** [`BenchCAD/cad_bench_722`](https://huggingface.co/datasets/BenchCAD/cad_bench_722) — 720 rows, single `train` split, the *diversified / substituted-parts* track of the BenchCAD benchmark.',
             '',
             '**Hardware:** RTX 4080 SUPER (16 GB).',
             '',
             '**Branch:** `eval/cad-bench-722`',
             '',
             '---',
             '',
             '## 1. Headline (greedy, single attempt)',
             '',
             f'| {"model":<32} | {"input":<28} | {"exec":>6} | {"mean IoU":>8} | {"mean CD":>9} |',
             f'|{"-"*34}|{"-"*30}|{"-"*8}|{"-"*10}|{"-"*11}|']
    inputs = {
        'cadrille_rl':         'pc + composite_png',
        'cadevolve_rl1':       '8-view 476×952 axis-coloured',
        'qwen25vl_3b_zs':      'composite_png 268×268',
        'cadrille_qwen3vl_v3': 'composite_png 268×268 (Qwen3-VL)',
    }
    for slug in MODEL_ORDER:
        s = _summarize_metadata(EVAL_ROOT / slug / 'metadata.jsonl')
        if not s: continue
        lines.append(f'| {MODEL_LABEL[slug]:<32} | {inputs[slug]:<28} | '
                     f'{_fmt_pct(s["exec_rate"]):>6} | '
                     f'{_fmt_iou(s["mean_iou"]):>8} | '
                     f'{_fmt_cd(s["mean_cd"]):>9} |')
    lines.append('')
    lines.append('### Per-difficulty (exec / mean IoU)')
    lines.append('')
    lines.append(f'| {"model":<32} | {"easy":<14} | {"medium":<14} | {"hard":<14} |')
    lines.append(f'|{"-"*34}|{"-"*16}|{"-"*16}|{"-"*16}|')
    for slug in MODEL_ORDER:
        s = _summarize_metadata(EVAL_ROOT / slug / 'metadata.jsonl')
        if not s: continue
        cells = []
        for d in ('easy', 'medium', 'hard'):
            ds = s.get('by_difficulty', {}).get(d, {})
            if ds.get('mean_iou') is not None:
                cells.append(f'{_fmt_pct(ds["exec_rate"])} / {ds["mean_iou"]:.3f}')
            else:
                cells.append(f'{_fmt_pct(ds.get("exec_rate", 0))} / —')
        lines.append(f'| {MODEL_LABEL[slug]:<32} | {cells[0]:<14} | '
                     f'{cells[1]:<14} | {cells[2]:<14} |')

    # ── 2. IoU-24 rotation rescue ──────────────────────────────────────
    lines.extend(['', '## 2. IoU-24 rotation rescue', '',
                  '`Δ = mean(iou_24 − iou)` over paired cases. `pct_rot_win` = '
                  'fraction of cases where a non-identity rotation beat the '
                  'identity, i.e. correct shape but oriented wrong.',
                  '',
                  f'| {"model":<32} | {"n_paired":>8} | {"mean iou":>8} | '
                  f'{"mean iou_24":>11} | {"Δ":>7} | {"pct_rot_win":>11} |',
                  f'|{"-"*34}|{"-"*10}|{"-"*10}|{"-"*13}|{"-"*9}|{"-"*13}|'])
    for slug in MODEL_ORDER:
        s = _summarize_metadata_24(EVAL_ROOT / slug / 'metadata_24.jsonl')
        if not s: continue
        lines.append(f'| {MODEL_LABEL[slug]:<32} | {s["n_paired"]:>8} | '
                     f'{_fmt_iou(s["mean_iou"]):>8} | '
                     f'{_fmt_iou(s["mean_iou_24"]):>11} | '
                     f'{s["mean_delta"]:>+7.4f} | '
                     f'{_fmt_pct(s["pct_rot_win"]):>11} |')

    # ── 3. Distribution-level metrics ──────────────────────────────────
    dist_path = EVAL_ROOT / 'distribution_metrics.json'
    if dist_path.exists():
        d = json.loads(dist_path.read_text())
        lines.extend(['', '## 3. Distribution-level metrics',
                      '', 'Computed against the full 720 GT image distribution. '
                      'FID / KID lower = better; CLIP R-Precision higher = better.',
                      '',
                      f'| {"model":<32} | {"n_pred":>6} | {"FID":>8} | '
                      f'{"KID":>9} | {"R@1":>6} | {"R@5":>6} | {"R@10":>6} |',
                      f'|{"-"*34}|{"-"*8}|{"-"*10}|{"-"*11}|{"-"*8}|{"-"*8}|{"-"*8}|'])
        for slug in MODEL_ORDER:
            md = d['models'].get(slug)
            if not md: continue
            rp = md.get('clip_r_precision', {})
            lines.append(f'| {MODEL_LABEL[slug]:<32} | '
                         f'{md.get("n_pred_rendered", 0):>6} | '
                         f'{md.get("fid_vs_full_720_gt", 0):>8.2f} | '
                         f'{md.get("kid_mean", 0):>9.5f} | '
                         f'{rp.get("r_at_1", 0):>6.3f} | '
                         f'{rp.get("r_at_5", 0):>6.3f} | '
                         f'{rp.get("r_at_10", 0):>6.3f} |')

    # ── 4. Out-of-distribution: DeepCAD + Fusion360 ────────────────────
    for ds_name in ('deepcad', 'fusion360'):
        ds_root = REPO / 'eval_outputs' / f'{ds_name}_n300'
        if not ds_root.exists(): continue
        lines.extend(['', f'## 4{"a" if ds_name=="deepcad" else "b"}. '
                      f'Out-of-distribution: {ds_name.capitalize()} '
                      f'(300 samples, seed=42)',
                      '',
                      f'| {"model":<32} | {"n":>4} | {"exec":>6} | '
                      f'{"mean IoU":>8} | {"mean CD":>9} |',
                      f'|{"-"*34}|{"-"*6}|{"-"*8}|{"-"*10}|{"-"*11}|'])
        for slug in MODEL_ORDER:
            s = _summarize_metadata(ds_root / slug / 'metadata.jsonl')
            if not s: continue
            lines.append(f'| {MODEL_LABEL[slug]:<32} | {s["n"]:>4} | '
                         f'{_fmt_pct(s["exec_rate"]):>6} | '
                         f'{_fmt_iou(s["mean_iou"]):>8} | '
                         f'{_fmt_cd(s["mean_cd"]):>9} |')

    # ── 5. Pointers to artifacts ───────────────────────────────────────
    lines.extend(['', '---', '',
                  '## Artifacts',
                  '',
                  '```',
                  'eval_outputs/cad_bench_722/',
                  '  cadrille_rl/      metadata.jsonl  metadata_24.jsonl  720 × <stem>.py',
                  '  cadevolve_rl1/    metadata.jsonl  metadata_24.jsonl  720 × <stem>.py',
                  '  qwen25vl_3b_zs/   metadata.jsonl  metadata_24.jsonl  720 × <stem>.py',
                  '  cadrille_qwen3vl_v3/  metadata.jsonl  metadata_24.jsonl  720 × <stem>.py',
                  '  summary.json                  — model-level IoU/CD',
                  '  summary_iou_24.json           — IoU-24 rescue summary',
                  '  distribution_metrics.json     — FID / KID / CLIP R-Precision',
                  '  metrics_per_case_full.json    — per-case Fs / DINO / LPIPS / SSIM / PSNR',
                  '  iou_vs_iou24/{report.md, scatter.png, histogram.png, rotation_dist.png}',
                  '  full_case_grids/cases_NNNN-NNNN.png × 15  — visual grid, 4 model columns',
                  '  RESULTS.md                    — this file',
                  'eval_outputs/deepcad_n300/<model>/metadata.jsonl   — OOD sample',
                  'eval_outputs/fusion360_n300/<model>/metadata.jsonl — OOD sample',
                  '```',
                  '',
                  '## How to reproduce',
                  '',
                  '```bash',
                  'set -a; source .env; eval "$(grep \'^export DISCORD\' ~/.bashrc)"; set +a',
                  '',
                  '# 1. cad_bench_722 (greedy, all 4 models — already wired in eval/bench.py)',
                  'bash scripts/eval_cad_bench_722.sh',
                  '',
                  '# 2. IoU-24 rotation rescore on the resulting metadata',
                  'bash scripts/run_rescore_iou_24.sh',
                  '',
                  '# 3. Extended per-case metrics (F-score / DINO / LPIPS / SSIM)',
                  'uv run python research/3d_similarity/compute_full_metrics.py',
                  '',
                  '# 4. Distribution-level (FID / KID / CLIP R-Precision)',
                  'uv run python research/3d_similarity/score_distribution.py',
                  '',
                  '# 5. IoU vs IoU-24 analysis (figures + report.md)',
                  'uv run python research/3d_similarity/analyze_iou_vs_iou24.py',
                  '',
                  '# 6. Full 720-case visual grid (15 PNG pages)',
                  'uv run python research/3d_similarity/build_full_grid.py',
                  '',
                  '# 7. OOD: DeepCAD + Fusion360 sampled n=300 each',
                  'bash scripts/run_stl_eval_all.sh',
                  '',
                  '# 8. Rebuild this markdown',
                  'uv run python scripts/analysis/build_summary_md.py',
                  '```',
                  ''])

    md = '\n'.join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.snapshot).write_text(md)
    print(f'Wrote {args.out} ({len(md)} chars)')
    print(f'Wrote {args.snapshot}')


if __name__ == '__main__':
    main()
