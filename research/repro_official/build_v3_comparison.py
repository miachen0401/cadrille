"""Build the final cad_bench_722 + OOD comparison after CADEvolve v3 lands.

Aggregates all 4 baselines (Cadrille-rl repro, CADEvolve-rl1 v3, Cadrille-Q3VL-v3,
Qwen2.5-VL-3B-zs) on every available metric:

  cad_bench_722 (720):
    - IoU, IoU-24, CD (mean over exec_ok)
    - exec rate
    - essential_pass + ess_cw + F1 + F1_cw
    - per-difficulty IoU breakdown (easy / medium / hard)
    - per-family pass-rate (top 20)
    - distribution metrics (FID, KID, R-Precision) — if scored

  OOD (DeepCAD-300, Fusion360-300):
    - IoU, CD, exec rate

Posts a single markdown report + a copy to Discord.

Usage:
    set -a; source .env; set +a
    .venv/bin/python research/repro_official/build_v3_comparison.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


MODELS = [
    ('cadrille_rl_repro',    'Cadrille-rl (4.50.3 repro)'),
    ('cadevolve_rl1',        'CADEvolve-rl1 (v3 fixed)'),
    ('cadrille_qwen3vl_v3',  'Cadrille-Q3VL-v3 (ours)'),
    ('qwen25vl_3b_zs',       'Qwen2.5-VL-3B (zs)'),
]
META = {
    'cadrille_rl_repro':    REPO / 'eval_outputs/repro_official/cad_bench_722_full/metadata.jsonl',
    'cadevolve_rl1':        REPO / 'eval_outputs/cad_bench_722/cadevolve_rl1/metadata.jsonl',
    'cadrille_qwen3vl_v3':  REPO / 'eval_outputs/cad_bench_722/cadrille_qwen3vl_v3/metadata.jsonl',
    'qwen25vl_3b_zs':       REPO / 'eval_outputs/cad_bench_722/qwen25vl_3b_zs/metadata.jsonl',
}
META_24 = {
    'cadrille_rl_repro':    None,  # no iou_24 yet for paper-repro
    'cadevolve_rl1':        REPO / 'eval_outputs/cad_bench_722/cadevolve_rl1/metadata_24.jsonl',
    'cadrille_qwen3vl_v3':  REPO / 'eval_outputs/cad_bench_722/cadrille_qwen3vl_v3/metadata_24.jsonl',
    'qwen25vl_3b_zs':       REPO / 'eval_outputs/cad_bench_722/qwen25vl_3b_zs/metadata_24.jsonl',
}
ESS_PATH = REPO / 'eval_outputs/cad_bench_722/essential_ops.json'
DIST_PATH = REPO / 'eval_outputs/cad_bench_722/distribution_metrics.json'

# OOD eval (the OLD buggy CADEvolve numbers; we override CADEvolve with v3)
OOD = {
    'deepcad-300': {
        'cadrille_rl_repro':   REPO / 'eval_outputs/repro_official/deepcad_test_mesh_n300/score.txt',
        'cadevolve_rl1':       REPO / 'eval_outputs/repro_official/deepcad_n300/cadevolve_v3/metadata.jsonl',
        'cadrille_qwen3vl_v3': REPO / 'eval_outputs/deepcad_n300/cadrille_qwen3vl_v3/metadata.jsonl',
        'qwen25vl_3b_zs':      REPO / 'eval_outputs/deepcad_n300/qwen25vl_3b_zs/metadata.jsonl',
    },
    'fusion360-300': {
        'cadrille_rl_repro':   REPO / 'eval_outputs/repro_official/fusion360_test_mesh_n300/score.txt',
        'cadevolve_rl1':       REPO / 'eval_outputs/repro_official/fusion360_n300/cadevolve_v3/metadata.jsonl',
        'cadrille_qwen3vl_v3': REPO / 'eval_outputs/fusion360_n300/cadrille_qwen3vl_v3/metadata.jsonl',
        'qwen25vl_3b_zs':      REPO / 'eval_outputs/fusion360_n300/qwen25vl_3b_zs/metadata.jsonl',
    },
}


def _load_jsonl(p):
    if p is None or not Path(p).exists():
        return []
    return [json.loads(l) for l in open(p)]


def _summary(rs):
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    return {
        'n': len(rs),
        'exec_pct': len(ok) / len(rs) * 100 if rs else 0,
        'mean_iou': sum(ious) / len(ious) if ious else None,
        'mean_cd':  sum(cds) / len(cds) if cds else None,
    }


def _summary_24(rs24):
    ok = [r for r in rs24 if r.get('error_type') == 'success']
    iou_24s = [r['iou_24'] for r in ok if r.get('iou_24') is not None]
    return sum(iou_24s) / len(iou_24s) if iou_24s else None


def _per_difficulty(rs):
    out = {}
    for diff in ('easy', 'medium', 'hard'):
        subset = [r for r in rs
                  if r.get('error_type') == 'success'
                  and r.get('difficulty') == diff
                  and r.get('iou') is not None]
        if subset:
            out[diff] = sum(r['iou'] for r in subset) / len(subset)
    return out


def _ood_score_txt(p):
    """Parse score.txt (Cadrille's evaluate.py output)."""
    if not p.exists(): return None
    text = p.read_text()
    out = {'mean_iou': None, 'mean_cd': None, 'n': None}
    for line in text.splitlines():
        line = line.strip()
        if 'mean iou' in line.lower():
            try: out['mean_iou'] = float(line.split(':')[-1].strip().split()[0])
            except: pass
        elif 'mean cd' in line.lower():
            try: out['mean_cd'] = float(line.split(':')[-1].strip().split()[0])
            except: pass
    return out


def _post_to_discord(content, attachments=None):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  (no DISCORD_WEBHOOK_URL — skip)')
        return False
    if not attachments:
        data = json.dumps({'content': content}).encode()
        req = urllib.request.Request(url, data=data,
            headers={'Content-Type': 'application/json',
                     'User-Agent': 'cadevolve-v3-compare/1.0'})
    else:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        def w(s): body.write(s.encode())
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
        w('Content-Type: application/json\r\n\r\n')
        w(json.dumps({'content': content}) + '\r\n')
        for i, p in enumerate(attachments):
            ct = 'image/png' if p.suffix.lower() == '.png' else 'text/markdown'
            w(f'--{boundary}\r\nContent-Disposition: form-data; '
              f'name="files[{i}]"; filename="{p.name}"\r\n')
            w(f'Content-Type: {ct}\r\n\r\n')
            body.write(p.read_bytes()); w('\r\n')
        w(f'--{boundary}--\r\n')
        req = urllib.request.Request(url, data=body.getvalue(), headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'User-Agent': 'cadevolve-v3-compare/1.0',
        })
    try:
        urllib.request.urlopen(req, timeout=30).read()
        return True
    except Exception as e:
        print(f'  Discord post failed: {e}')
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(REPO / 'eval_outputs/cad_bench_722/v3_comparison.md'))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    # ── cad_bench_722 ───────────────────────────────────────────────────────
    rows = {}
    for slug, label in MODELS:
        rs = _load_jsonl(META.get(slug))
        rs24 = _load_jsonl(META_24.get(slug))
        rows[slug] = {
            'label': label,
            **_summary(rs),
            'mean_iou_24': _summary_24(rs24),
            'per_difficulty': _per_difficulty(rs),
        }

    # essential_ops
    ess = json.loads(ESS_PATH.read_text()) if ESS_PATH.exists() else None

    # distribution metrics
    dist = json.loads(DIST_PATH.read_text()) if DIST_PATH.exists() else None

    # OOD
    ood_rows = {}
    for ds_name, paths in OOD.items():
        ood_rows[ds_name] = {}
        for slug, _ in MODELS:
            p = paths.get(slug)
            if p is None or not Path(p).exists():
                ood_rows[ds_name][slug] = None
                continue
            if Path(p).suffix == '.txt':
                s = _ood_score_txt(Path(p))
            else:
                rs = _load_jsonl(p)
                s = _summary(rs)
            ood_rows[ds_name][slug] = s

    # ── Markdown ────────────────────────────────────────────────────────────
    L = ['# cad_bench_722 + OOD — full comparison after CADEvolve v3 fix', '']
    L.append('Generated by `research/repro_official/build_v3_comparison.py`. '
             'CADEvolve-rl1 here is the **v3 fixed-setup** run (official Plotter '
             '+ pre-normalized STL + image-only prompt + max_new_tokens=4000). '
             'Old buggy results are at `eval_outputs/.../cadevolve_rl1_buggy/`.')
    L.append('')

    # 1. Headline
    L.append('## 1. Headline (cad_bench_722, greedy)')
    L.append('')
    L.append('| model                       | n  | exec   | mean IoU | mean IoU-24 | mean CD |')
    L.append('|-----------------------------|----|--------|----------|-------------|---------|')
    for slug, _ in MODELS:
        r = rows[slug]
        iou_24 = (f'{r["mean_iou_24"]:.4f}' if r['mean_iou_24'] is not None else '—')
        L.append(f'| {r["label"]:<27} | {r["n"]:>3} | {r["exec_pct"]:>5.1f}% | '
                 f'{r["mean_iou"]:.4f}   | {iou_24:<9}   | {r["mean_cd"]:.4f}  |')
    L.append('')

    # 2. Per-difficulty
    L.append('## 2. Per-difficulty mean IoU (cad_bench_722)')
    L.append('')
    L.append('| model | easy | medium | hard |')
    L.append('|-------|------|--------|------|')
    for slug, _ in MODELS:
        r = rows[slug]
        pd = r['per_difficulty']
        L.append(f'| {r["label"]} | '
                 f'{pd.get("easy", float("nan")):.3f} | '
                 f'{pd.get("medium", float("nan")):.3f} | '
                 f'{pd.get("hard", float("nan")):.3f} |')
    L.append('')

    # 3. Essential-ops
    if ess:
        L.append('## 3. Essential-ops + feature-F1 (cad_bench_722)')
        L.append('')
        L.append('`ess` = pass / (pass+fail) on exec_ok subset; '
                 '`ess_cw` = pass / 720 (coverage-weighted, fairer when exec rates differ); '
                 '`F1` = exec-only mean; `F1_cw` = coverage-weighted (missing → 0).')
        L.append('')
        L.append('| model | n_pred | ess | ess_cw | F1 | F1_cw |')
        L.append('|-------|--------|-----|--------|-----|------|')
        for slug, _ in MODELS:
            d = ess['models'].get(slug)
            if not d:
                L.append(f'| {rows[slug]["label"]} | — | — | — | — | — |')
                continue
            ep = (f'{d["pct_essential_pass"]*100:.1f}%' if d['pct_essential_pass'] is not None else '—')
            ep_cw = f'{d.get("pct_essential_pass_cw", 0)*100:.2f}%'
            ff = (f'{d["mean_feature_f1"]:.3f}' if d['mean_feature_f1'] is not None else '—')
            ff_cw = f'{d.get("mean_feature_f1_cw", 0):.3f}'
            L.append(f'| {rows[slug]["label"]} | {d["n_with_pred"]} | '
                     f'{ep} ({d["n_pass"]}/{d["n_pass"]+d["n_fail"]}) | '
                     f'{ep_cw} | {ff} | {ff_cw} |')
        L.append('')

    # 4. Distribution metrics
    if dist:
        L.append('## 4. Distribution metrics (cad_bench_722)')
        L.append('')
        L.append('| model | n_pred | FID ↓ | KID ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |')
        L.append('|-------|--------|-------|-------|-------|-------|--------|')
        for slug, _ in MODELS:
            d = dist.get('models', {}).get(slug)
            if not d:
                L.append(f'| {rows[slug]["label"]} | — | — | — | — | — | — |')
                continue
            L.append(f'| {rows[slug]["label"]} | {d.get("n_pred", "—")} | '
                     f'{d.get("fid", float("nan")):.2f} | '
                     f'{d.get("kid", float("nan")):.4f} | '
                     f'{d.get("r1", 0):.3f} | '
                     f'{d.get("r5", 0):.3f} | '
                     f'{d.get("r10", 0):.3f} |')
        L.append('')

    # 5. OOD
    L.append('## 5. Out-of-distribution: DeepCAD-300, Fusion360-300')
    L.append('')
    for ds_name, ds_rows in ood_rows.items():
        L.append(f'### {ds_name}')
        L.append('')
        L.append('| model | n | exec | mean IoU | mean CD |')
        L.append('|-------|----|------|----------|---------|')
        for slug, _ in MODELS:
            r = ds_rows.get(slug)
            if r is None:
                L.append(f'| {rows[slug]["label"]} | — | — | — | — |')
                continue
            n = r.get('n') or '—'
            exec_p = f'{r["exec_pct"]:.1f}%' if r.get('exec_pct') is not None else '—'
            mi = f'{r["mean_iou"]:.4f}' if r.get('mean_iou') is not None else '—'
            mc = f'{r["mean_cd"]:.4f}' if r.get('mean_cd') is not None else '—'
            L.append(f'| {rows[slug]["label"]} | {n} | {exec_p} | {mi} | {mc} |')
        L.append('')

    # 5b. Per-family breakdown (top 12 by sample count)
    L.append('## 5b. Per-family mean IoU (top 12 families by count)')
    L.append('')
    fam_count = defaultdict(int)
    for r in _load_jsonl(META['cadrille_qwen3vl_v3']):
        if r.get('family'):
            fam_count[r['family']] += 1
    top_fams = sorted(fam_count.items(), key=lambda kv: -kv[1])[:12]
    L.append('| family | n | ' + ' | '.join(rows[s]['label'].split()[0] for s, _ in MODELS) + ' |')
    L.append('|--------|----|' + '|'.join(['------'] * len(MODELS)) + '|')
    by_fam = {slug: defaultdict(list) for slug, _ in MODELS}
    for slug, _ in MODELS:
        for r in _load_jsonl(META.get(slug)):
            if r.get('error_type') == 'success' and r.get('iou') is not None and r.get('family'):
                by_fam[slug][r['family']].append(r['iou'])
    for fam, n in top_fams:
        cells = []
        for slug, _ in MODELS:
            xs = by_fam[slug].get(fam, [])
            cells.append(f'{sum(xs)/len(xs):.3f}' if xs else '—')
        L.append(f'| {fam} | {n} | ' + ' | '.join(cells) + ' |')
    L.append('')

    # 5c. Failure modes + code length
    L.append('## 5c. Failure modes + code length (cad_bench_722)')
    L.append('')
    L.append('| model | n | exec% | err_breakdown | code_len med / p90 |')
    L.append('|-------|----|-------|--------------|---------------------|')
    for slug, _ in MODELS:
        rs = _load_jsonl(META.get(slug))
        from collections import Counter as _C
        err = _C(r.get('error_type', '?') for r in rs)
        n_ok = err.get('success', 0)
        err_top = ', '.join(f'{k}:{v}' for k, v in err.most_common(4))
        code_lens = [r.get('code_len') for r in rs if r.get('code_len') is not None]
        if code_lens:
            cs = sorted(code_lens)
            cl = f'{cs[len(cs)//2]} / {cs[int(0.9*len(cs))]}'
        else:
            cl = '—'
        L.append(f'| {rows[slug]["label"]} | {len(rs)} | '
                 f'{n_ok/len(rs)*100 if rs else 0:.1f}% | '
                 f'`{err_top[:60]}` | {cl} |')
    L.append('')

    # 5d. Op-distribution alignment
    L.append('## 5d. Op-distribution alignment (cad_bench_722, ess samples)')
    L.append('')
    L.append('Mean (gen ops ∩ gt ops) and (gen ops − gt ops) sizes per case '
             '— larger overlap is better, larger spurious is worse.')
    L.append('')
    L.append('| model | mean overlap | mean spurious | mean missing |')
    L.append('|-------|--------------|---------------|--------------|')
    if ess:
        for slug, _ in MODELS:
            d = ess['models'].get(slug)
            if not d: continue
            ov, sp, mi = [], [], []
            for c in d['per_case']:
                gen = set(c.get('gen_ops') or [])
                gt  = set(c.get('gt_ops') or [])
                ov.append(len(gen & gt))
                sp.append(len(gen - gt))
                mi.append(len(gt - gen))
            n = len(d['per_case'])
            if n:
                L.append(f'| {rows[slug]["label"]} | '
                         f'{sum(ov)/n:.2f} | '
                         f'{sum(sp)/n:.2f} | '
                         f'{sum(mi)/n:.2f} |')
    L.append('')

    # 6. Why CADEvolve is strong
    L.append('## 6. Why CADEvolve is strong (architectural diff vs ours)')
    L.append('')
    L.append('After fixing the inference setup, CADEvolve-rl1 jumps ahead of '
             'Cadrille-Q3VL-v3 on every IoU metric. Three architectural '
             'choices likely drive the gap:')
    L.append('')
    L.append('1. **8-view 476×952 input** vs our 4-view 268×268. ~6× more '
             'pixels and 2× more views → much richer 3D evidence.')
    L.append('2. **Coordinate-encoded green channel** (each ortho view paints '
             'the depth axis as `G = 255 * coord ∈ [0,1]`). This bakes in '
             'an explicit spatial reference that the model cannot learn from '
             'shading alone.')
    L.append('3. **RL-finetuned directly on IoU** (per CADEvolve paper). Our '
             'Q3VL-v3 is SFT-only on a wider (and noisier) 100k mix.')
    L.append('')
    L.append('Cheap experiments to test (1)+(2) on our model: (a) re-render '
             'the SFT corpus through the official Plotter (8-view 476×952) '
             'and continue SFT; (b) add a coord-channel-only augmentation '
             'during SFT.')
    L.append('')

    out_path = Path(args.out)
    out_path.write_text('\n'.join(L))
    print(f'Wrote {out_path}')

    if args.discord:
        # Top-level summary in Discord embed
        msg = ['🏆 **cad_bench_722 + OOD — full comparison after CADEvolve v3 fix**',
               '',
               '```',
               f'{"model":<28} {"IoU":>6} {"IoU-24":>7} {"ess_cw":>7} {"F1_cw":>6}',
               '-' * 60]
        for slug, _ in MODELS:
            r = rows[slug]
            d = (ess['models'].get(slug) if ess else None) or {}
            iou_str = f'{r["mean_iou"]:.3f}' if r.get('mean_iou') is not None else '—'
            iou24_str = f'{r["mean_iou_24"]:.3f}' if r.get('mean_iou_24') is not None else '—'
            ep_cw = f'{d.get("pct_essential_pass_cw", 0)*100:.1f}%' if d else '—'
            ff_cw = f'{d.get("mean_feature_f1_cw", 0):.3f}' if d else '—'
            msg.append(f'{r["label"]:<28} {iou_str:>6} {iou24_str:>7} {ep_cw:>7} {ff_cw:>6}')
        msg.append('```')
        msg.append('Full report attached.')
        ok = _post_to_discord('\n'.join(msg), attachments=[out_path])
        print(f'  Discord: {"sent" if ok else "FAILED"}')


if __name__ == '__main__':
    main()
