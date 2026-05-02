"""4-baseline comparison table — uses existing cached metrics only.

Reads from:
  - metadata.jsonl per model (iou, cd, error_type)
  - essential_ops.json (essential_score fractional, feature_f1)

Computes:
  - CD-score = 1 / (1 + cd) — algebraic, [0, 1], no fresh exec needed
  - HD / HD-score — left as TBD (would need full re-exec; deferred)
  - total = mean of (IoU_cw, CD_score_cw, feature_score, ess_score)

OOD reads pre-computed metadata.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

REPO = Path('/home/hula0401/Projects/cadrille')
EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'

MODELS = [
    ('cadrille_rl_repro',    'Cadrille-rl'),
    ('cadevolve_rl1',        'CADEvolve v3'),
    ('cadrille_qwen3vl_v3',  'Q3VL (ours)'),
    ('qwen25vl_3b_zs',       'Qwen-zs'),
]
META_PATH = {
    'cadrille_rl_repro':   REPO / 'eval_outputs/repro_official/cad_bench_722_full/metadata.jsonl',
    'cadevolve_rl1':       EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'cadrille_qwen3vl_v3': EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
    'qwen25vl_3b_zs':      EVAL_ROOT / 'qwen25vl_3b_zs' / 'metadata.jsonl',
}
OOD = {
    # Cadrille-rl uses the paper-repro IoU/CD from score.txt (cadquery worker
    # pool deadlocked when re-scoring through our pipeline; the paper-repro
    # numbers are canonical for that model — 0.915 / 0.838 on DeepCAD/Fusion360
    # IoU). The other 3 models are scored through our v3 pipeline.
    'DeepCAD-300': {
        'cadrille_rl_repro':   REPO / 'eval_outputs/repro_official/deepcad_test_mesh_n300/score.txt',
        'cadevolve_rl1':       REPO / 'eval_outputs/deepcad_n300/cadevolve_rl1/metadata.jsonl',
        'cadrille_qwen3vl_v3': REPO / 'eval_outputs/deepcad_n300/cadrille_qwen3vl_v3/metadata.jsonl',
        'qwen25vl_3b_zs':      REPO / 'eval_outputs/deepcad_n300/qwen25vl_3b_zs/metadata.jsonl',
    },
    'Fusion360-300': {
        'cadrille_rl_repro':   REPO / 'eval_outputs/repro_official/fusion360_test_mesh_n300/score.txt',
        'cadevolve_rl1':       REPO / 'eval_outputs/fusion360_n300/cadevolve_rl1/metadata.jsonl',
        'cadrille_qwen3vl_v3': REPO / 'eval_outputs/fusion360_n300/cadrille_qwen3vl_v3/metadata.jsonl',
        'qwen25vl_3b_zs':      REPO / 'eval_outputs/fusion360_n300/qwen25vl_3b_zs/metadata.jsonl',
    },
}


def _post(content, attachment):
    url = open(REPO / '.env').read().split('DISCORD_WEBHOOK_URL=')[1].split()[0]
    boundary = uuid.uuid4().hex; body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
    w('Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    ct = 'image/png' if attachment.suffix == '.png' else 'text/markdown'
    w(f'--{boundary}\r\nContent-Disposition: form-data; '
      f'name="file"; filename="{attachment.name}"\r\n')
    w(f'Content-Type: {ct}\r\n\r\n')
    body.write(attachment.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-comparison-simple/1.0',
    })
    urllib.request.urlopen(req, timeout=30).read()


def _ood(p):
    """Read summary metrics. Supports both metadata.jsonl and the upstream
    score.txt format which crams multiple key:value pairs on one line:

        mean iou: 0.915 median cd: 0.168
        skip: 0 ir: 0.00 mean cd: 0.515

    Use a regex that grabs the float immediately after each named field —
    naive `line.split(':')[-1]` reads the LAST value (median cd 0.168) and
    treats it as the iou. That bug produced IoU=0.168 for the Cadrille-rl
    paper-repro DeepCAD eval (true value 0.915)."""
    import re
    p = Path(p)
    if not p.exists(): return None
    if p.suffix == '.txt':
        text = p.read_text()
        m_iou = re.search(r'mean iou:\s*([0-9.]+)', text)
        m_cd  = re.search(r'\bmean cd:\s*([0-9.]+)', text)
        # `mean cd:` appears multiple times in upstream output (one per
        # ir-skip threshold); pick the FIRST one (skip=0, no filtering).
        return {
            'n': None,
            'exec_pct': None,
            'iou': float(m_iou.group(1)) if m_iou else None,
            'cd':  float(m_cd.group(1))  if m_cd  else None,
        }
    rs = [json.loads(l) for l in open(p)]
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    return {
        'n': len(rs),
        'exec_pct': len(ok)/len(rs)*100 if rs else 0,
        'iou': sum(ious)/len(ious) if ious else None,
        'cd':  sum(cds)/len(cds)   if cds  else None,
    }


def main():
    # Load metadata
    metas = {}
    for slug, _ in MODELS:
        metas[slug] = {}
        for line in open(META_PATH[slug]):
            try: r = json.loads(line); metas[slug][r['stem']] = r
            except Exception: pass

    ess = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    n_total = ess.get('n_total', 720)

    # Per-model aggregates
    rows = {}
    for slug, label in MODELS:
        d = ess['models'].get(slug, {})
        n_pred = len(metas[slug])
        ok = [r for r in metas[slug].values() if r.get('error_type') == 'success']
        ious = [r['iou'] for r in ok if r.get('iou') is not None]
        cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
        # CD-score = 1 / (1 + cd) per case → coverage-weighted mean
        cd_scores = [1.0 / (1.0 + r['cd']) for r in ok if r.get('cd') is not None]
        rows[slug] = {
            'label':   label,
            'n_pred':  n_pred,
            'exec_pct': len(ok) / n_pred * 100 if n_pred else 0,
            'iou_raw': sum(ious)/len(ious) if ious else float('nan'),  # exec-only mean
            'cd_raw':  sum(cds)/len(cds)   if cds  else float('nan'),  # exec-only mean
            # coverage-weighted scores (denom = n_total = 720)
            'iou_cw':       sum(ious) / n_total,
            'cd_score_cw':  sum(cd_scores) / n_total,  # 1/(1+cd), missing→0
            'ess_score':    d.get('mean_essential_score_cw', 0),
            'feature_score': d.get('mean_feature_f1_cw', 0),
        }
        # total = mean of 4 [0,1] scores (HD-score skipped)
        scores = [rows[slug]['iou_cw'], rows[slug]['cd_score_cw'],
                  rows[slug]['ess_score'], rows[slug]['feature_score']]
        rows[slug]['total'] = sum(scores) / len(scores)

    # OOD
    ood_rows = {ds: {slug: _ood(p) for slug, p in paths.items()}
                for ds, paths in OOD.items()}

    # Markdown
    L = ['# 4-baseline comparison — cad_bench_722 + OOD',
         '',
         'Computed from existing per-model metadata + `common/essential_ops` scoring. '
         'No fresh STL re-exec — IoU and CD come from `metadata.jsonl`, '
         'ess_score and feature_score from `essential_ops.json` (post-Cadance#10 spec). '
         'CD-score is derived algebraically (`1 / (1 + cd)`); HD/HD-score require fresh '
         'point-sampling pass and are deferred (see TODO note below).',
         '',
         '## cad_bench_722 (n = 720)',
         '',
         '| model | exec | IoU | CD | CD-score | feature_score | ess_score | **total** |',
         '|-------|------|-----|----|----------|---------------|-----------|-----------|']
    for slug, _ in MODELS:
        r = rows[slug]
        L.append(f'| {r["label"]} | {r["exec_pct"]:.1f}% | '
                 f'{r["iou_cw"]:.3f} | {r["cd_raw"]:.4f} | {r["cd_score_cw"]:.3f} | '
                 f'{r["feature_score"]:.3f} | {r["ess_score"]:.3f} | '
                 f'**{r["total"]:.3f}** |')

    L += ['',
          '**Column meanings** (all coverage-weighted over 720 cases unless noted):',
          '- `exec` — fraction of preds that exec without error',
          '- `IoU` — voxel intersection-over-union, mean over all 720 cases (missing → 0)',
          '- `CD` — Chamfer Distance (raw, exec-only mean; **lower** is better)',
          '- `CD-score` = `mean(1 / (1 + cd))` over all 720 cases (missing → 0). Higher is better.',
          '- `feature_score` — mean `feature_f1` over `{chamfer, fillet, hole}` (Cadance#10 spec)',
          '- `ess_score` — mean fractional `essential_score` (Cadance#10 spec)',
          '- `total` = mean of `(IoU, CD-score, feature_score, ess_score)`',
          '',
          '**TODO** — Hausdorff Distance (HD) + HD-score. Computing HD requires re-execing all '
          '~2900 (gt, pred) pairs and sampling 8192 surface points each. The previous attempt '
          'hung in cadquery for some preds (worker pool deadlock). Will run separately.',
          '',
          '**Raw exec-only IoU** (for reference, in case the cw normalisation is misleading):']
    for slug, _ in MODELS:
        r = rows[slug]
        L.append(f'  - {r["label"]}: {r["iou_raw"]:.3f}')

    L += ['',
          '## DeepCAD-300 / Fusion360-300 test sets',
          '',
          '(Cadrille-rl was trained on DeepCAD + Fusion360 train, so these are '
          'in-distribution test sets *for that model*; for CADEvolve, Q3VL, '
          'and Qwen-zs they are out-of-distribution. Avoid the IID/OOD label '
          'in the headline — it depends on which model you ask.)',
          '',
          '⚠️ **CD scale notice**: Cadrille-rl numbers are from the upstream '
          '`evaluate.py` (paper-repro 4.50.3) which uses a different mesh '
          'normalisation than our v3 pipeline. **IoU is directly comparable** '
          '(both report voxel IoU); **CD is NOT** — Cadrille-rl CD is on a '
          '~250× larger scale. Treat CD across the Cadrille-rl row separately.',
          '',
          '| dataset | model | n | exec | IoU | CD |',
          '|---------|-------|----|------|-----|-----|']
    for ds_name, ds_rows in ood_rows.items():
        for slug, _ in MODELS:
            r = ds_rows.get(slug)
            if r is None:
                L.append(f'| {ds_name} | {rows[slug]["label"]} | — | — | — | — |')
                continue
            n = r.get('n') or '—'
            ex = f'{r["exec_pct"]:.1f}%' if r.get('exec_pct') is not None else '—'
            iou = f'{r["iou"]:.3f}' if r.get('iou') is not None else '—'
            cd  = f'{r["cd"]:.4f}'  if r.get('cd')  is not None else '—'
            L.append(f'| {ds_name} | {rows[slug]["label"]} | {n} | {ex} | {iou} | {cd} |')

    out = EVAL_ROOT / 'comparison_table.md'
    out.write_text('\n'.join(L))
    print(f'Wrote {out}')

    msg = ('📊 **4-baseline comparison — cad_bench_722 + OOD**\n'
           '\n'
           'cad_bench_722: IoU + CD + CD-score (`1/(1+cd)`) + feature_score + ess_score + total.\n'
           'All scores coverage-weighted (n=720). HD/HD-score deferred — '
           'previous compute attempt hung in cadquery worker pool.\n'
           '\n'
           'OOD: DeepCAD-300 + Fusion360-300, IoU + CD + exec.')
    _post(msg, out)
    print('  posted to Discord ✓')


if __name__ == '__main__':
    main()
