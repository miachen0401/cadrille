"""Step 0.5: dim_error sub-classification via STL bounding-box analysis.

For every low-IoU success case (IoU in (0, 0.70)) across all 8 combos, applies the
same rule-based classifier used in error_taxonomy_0321.md to identify dim_error cases,
then loads pred + GT STL to compare normalised bounding-box extents.

Sub-types after normalising both meshes to [-1, 1]³ (same as scoring pipeline):
  aspect_ratio  — sorted normalised extents differ by >0.15 (wrong proportions/shape)
  local_feat    — extents match (<0.15 diff) but IoU still low (holes/cutouts/features wrong)

Usage
-----
  python3 tools/analyze_dim_errors.py            # full run, all 8 combos
  python3 tools/analyze_dim_errors.py --workers 4
  python3 tools/analyze_dim_errors.py --combo deepcad_rl_img --iou-max 0.8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

# ── combos ───────────────────────────────────────────────────────────────────

_COMBOS = [
    ("deepcad_sft_img",   "deepcad",   "deepcad_test_mesh",   "sft"),
    ("deepcad_rl_img",    "deepcad",   "deepcad_test_mesh",   "rl"),
    ("deepcad_sft_pc",    "deepcad",   "deepcad_test_mesh",   "sft"),
    ("deepcad_rl_pc",     "deepcad",   "deepcad_test_mesh",   "rl"),
    ("fusion360_sft_img", "fusion360", "fusion360_test_mesh", "sft"),
    ("fusion360_rl_img",  "fusion360", "fusion360_test_mesh", "rl"),
    ("fusion360_sft_pc",  "fusion360", "fusion360_test_mesh", "sft"),
    ("fusion360_rl_pc",   "fusion360", "fusion360_test_mesh", "rl"),
]

# ── rule-based classifier (mirrors error_taxonomy_0321.md methodology) ───────

def _classify_code(code: str, iou: float) -> str:
    """Deterministic priority-ordered classifier."""
    nums = [float(x) for x in re.findall(r"\b\d+\.?\d*\b", code)]
    max_num = max(nums) if nums else 0

    extrude_vals = [float(x) for x in re.findall(r"\.extrude\((-?\d+\.?\d*)", code)]
    min_extrude = min(abs(v) for v in extrude_vals) if extrude_vals else 999

    n_union    = code.count(".union(")
    n_push     = len(re.findall(r"\.push\(\[", code))
    n_extrude  = len(re.findall(r"\.extrude\(", code))
    n_segments = code.count(".segment(") + code.count(".arc(")
    n_subtract = code.count("mode='s'")

    has_sketch      = ".sketch()" in code
    has_box_prim    = ".box(" in code and not has_sketch
    has_cyl_prim    = ".cylinder(" in code and not has_sketch

    # 1. degenerate
    if iou < 0.05:
        return "degenerate"
    if min_extrude <= 1.5 and max_num > 50 and iou < 0.15:
        return "degenerate"

    # 2. wrong_primitive
    if (has_box_prim or has_cyl_prim):
        return "wrong_primitive"

    # 3. partial_geom
    if n_extrude == 1 and n_union == 0:
        if n_segments > 8 and iou < 0.40:
            return "partial_geom"
        if n_subtract >= 2 and iou < 0.35:
            return "partial_geom"

    # 4. wrong_plane
    workplanes = re.findall(r"cq\.Workplane\('(\w+)'", code)
    non_xy = sum(1 for w in workplanes if w not in ("XY", "xy"))
    if non_xy > 0 and n_union == 0 and iou < 0.40 and n_segments < 8:
        return "wrong_plane"
    if min_extrude < 20 and max_num > 100 and iou < 0.20 and n_union == 0:
        return "wrong_plane"

    # 5. feature_count
    if n_union >= 4 and iou < 0.45:
        return "feature_count"
    if n_push > 5 and iou < 0.45:
        return "feature_count"

    # 6. dim_error (default)
    return "dim_error"


# ── per-case worker (spawned in process pool) ─────────────────────────────────

def _analyse_case(args: tuple) -> tuple:
    """Normalise both STLs, compare sorted extents, return sub-type."""
    combo, case_id, pred_stl, gt_stl, iou = args
    try:
        import trimesh
        pred = trimesh.load(str(pred_stl), force="mesh")
        gt   = trimesh.load(str(gt_stl),   force="mesh")

        if pred is None or gt is None or not isinstance(pred, trimesh.Trimesh):
            return combo, case_id, None
        if gt.is_empty or pred.is_empty:
            return combo, case_id, None

        def _norm(m):
            m = m.copy()
            m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
            ext = float(np.max(m.extents))
            if ext > 1e-6:
                m.apply_scale(2.0 / ext)
            return m

        pred_n = _norm(pred)
        gt_n   = _norm(gt)

        # sorted extents descending; after norm max = 2.0 for both
        pe = np.sort(pred_n.extents)[::-1]
        ge = np.sort(gt_n.extents)[::-1]

        # aspect ratio vectors [1.0, b/a, c/a]
        pred_asp = pe / pe[0]
        gt_asp   = ge / ge[0]
        asp_diff = float(np.max(np.abs(pred_asp - gt_asp)))

        # volume ratio in normalised space
        vol_ratio = float(pred_n.volume) / max(float(gt_n.volume), 1e-9)

        # sub-type
        if asp_diff >= 0.25:
            subtype = "aspect_ratio"
        elif asp_diff >= 0.12:
            subtype = "aspect_ratio_mild"
        else:
            subtype = "local_feat"

        return combo, case_id, {
            "subtype": subtype,
            "asp_diff": round(asp_diff, 4),
            "vol_ratio": round(vol_ratio, 4),
            "iou": round(iou, 4),
            "pred_asp": [round(x, 4) for x in pred_asp.tolist()],
            "gt_asp":   [round(x, 4) for x in gt_asp.tolist()],
        }
    except Exception as e:
        return combo, case_id, {"error": str(e)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combo", nargs="*", help="Subset of combos (default: all 8)")
    parser.add_argument("--iou-max", type=float, default=0.70,
                        help="Upper IoU bound for low-IoU filter (default: 0.70)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Process pool workers (default: 6)")
    parser.add_argument("--out", default="docs/analysis/dim_error_analysis_0321.md",
                        help="Output markdown file")
    args = parser.parse_args()

    analysis_dir = _REPO / "data" / "analysis"
    data_dir     = _REPO / "data"
    out_path     = _REPO / args.out

    combos = _COMBOS
    if args.combo:
        combos = [c for c in _COMBOS if c[0] in args.combo]

    # ── gather tasks ─────────────────────────────────────────────────────────
    # per_combo_taxonomy[combo] = {category: count}
    per_combo_taxonomy: dict[str, dict[str, int]] = {}
    # per_combo_dim[combo] = {subtype: count}
    per_combo_dim: dict[str, dict[str, int]] = {}
    per_combo_raw: dict[str, list[dict]] = {}  # raw result records

    all_tasks = []  # tasks for BB analysis

    print(f"{'='*64}")
    print("Phase 0.5 — dim_error sub-classification")
    print(f"{'='*64}\n")

    for combo, dataset, mesh_dir, model in combos:
        combo_dir = analysis_dir / combo
        meta_path = combo_dir / "metadata.jsonl"
        gt_dir    = data_dir / mesh_dir

        if not meta_path.exists():
            print(f"  SKIP {combo}: metadata.jsonl not found")
            continue

        # load metadata
        rows = []
        with open(meta_path) as f:
            for line in f:
                rows.append(json.loads(line))

        # filter: success + IoU in (0, iou_max)
        low_iou = [r for r in rows
                   if r["error_type"] == "success"
                   and r["iou"] is not None
                   and 0 < r["iou"] <= args.iou_max]

        print(f"  {combo}: {len(rows)} total  →  {len(low_iou)} low-IoU success cases")

        # apply rule-based classifier
        taxonomy_counts: dict[str, int] = {}
        dim_cases = []
        for r in low_iou:
            case_id  = r["case_id"]
            iou      = float(r["iou"])
            py_path  = combo_dir / f"{case_id}_pred.py"
            code = py_path.read_text() if py_path.exists() else ""
            cat = _classify_code(code, iou) if code else "no_code"
            taxonomy_counts[cat] = taxonomy_counts.get(cat, 0) + 1
            if cat == "dim_error":
                dim_cases.append((case_id, iou))

        per_combo_taxonomy[combo] = taxonomy_counts
        print(f"    taxonomy: {taxonomy_counts}")
        print(f"    → dim_error: {len(dim_cases)} cases for BB analysis")

        # queue BB analysis tasks
        per_combo_raw[combo] = []
        for case_id, iou in dim_cases:
            pred_stl = combo_dir / f"{case_id}_pred.stl"
            gt_stl   = gt_dir    / f"{case_id}.stl"
            if pred_stl.exists() and gt_stl.exists():
                all_tasks.append((combo, case_id, str(pred_stl), str(gt_stl), iou))

    print(f"\n  Total BB tasks: {len(all_tasks)}")
    print("  Running in process pool...\n")

    # ── run BB analysis ───────────────────────────────────────────────────────
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_analyse_case, t): t for t in all_tasks}
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  [{done}/{len(all_tasks)}] ...", flush=True)
            combo, case_id, result = fut.result()
            if result and "subtype" in result:
                per_combo_raw[combo].append(result)

    # ── aggregate per-combo dim sub-types ─────────────────────────────────────
    for combo in per_combo_raw:
        counts: dict[str, int] = {}
        for r in per_combo_raw[combo]:
            st = r["subtype"]
            counts[st] = counts.get(st, 0) + 1
        per_combo_dim[combo] = counts

    # ── additional stats per combo ────────────────────────────────────────────
    combo_stats: dict[str, dict] = {}
    for combo in per_combo_raw:
        records = per_combo_raw[combo]
        if not records:
            continue
        asp_diffs   = [r["asp_diff"]  for r in records]
        vol_ratios  = [r["vol_ratio"] for r in records]
        ious        = [r["iou"]       for r in records]
        combo_stats[combo] = {
            "n": len(records),
            "asp_diff_median": float(np.median(asp_diffs)),
            "asp_diff_p90":    float(np.percentile(asp_diffs, 90)),
            "vol_ratio_median":float(np.median(vol_ratios)),
            "iou_median":      float(np.median(ious)),
            "aspect_ratio_pct":   round(100 * sum(1 for r in records if r["subtype"] == "aspect_ratio") / len(records), 1),
            "aspect_ratio_mild_pct": round(100 * sum(1 for r in records if r["subtype"] == "aspect_ratio_mild") / len(records), 1),
            "local_feat_pct":  round(100 * sum(1 for r in records if r["subtype"] == "local_feat") / len(records), 1),
        }

    # ── volume bias: over vs under ────────────────────────────────────────────
    vol_bias: dict[str, dict] = {}
    for combo in per_combo_raw:
        records = per_combo_raw[combo]
        if not records:
            continue
        vols = [r["vol_ratio"] for r in records]
        over  = sum(1 for v in vols if v > 1.1)
        under = sum(1 for v in vols if v < 0.9)
        near  = len(vols) - over - under
        vol_bias[combo] = {
            "over_pct":  round(100 * over  / len(vols), 1),
            "near_pct":  round(100 * near  / len(vols), 1),
            "under_pct": round(100 * under / len(vols), 1),
            "median":    round(float(np.median(vols)), 3),
        }

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("RESULTS — dim_error sub-type breakdown")
    print(f"{'='*64}\n")

    # aggregate: SFT vs RL, img vs pc
    agg: dict[str, dict[str, int]] = {
        "sft_img": {}, "rl_img": {}, "sft_pc": {}, "rl_pc": {},
        "sft": {}, "rl": {}, "img": {}, "pc": {}, "all": {}
    }
    for combo, dataset, mesh_dir, model in combos:
        if combo not in per_combo_raw:
            continue
        modality = "img" if "img" in combo else "pc"
        records = per_combo_raw[combo]
        for r in records:
            st = r["subtype"]
            for key in [f"{model}_{modality}", model, modality, "all"]:
                agg[key][st] = agg[key].get(st, 0) + 1

    def _pct_row(d: dict) -> str:
        total = sum(d.values())
        if total == 0:
            return "(no data)"
        ar   = d.get("aspect_ratio", 0)
        arm  = d.get("aspect_ratio_mild", 0)
        lf   = d.get("local_feat", 0)
        return (f"aspect_ratio={ar}({ar*100//total}%)  "
                f"aspect_mild={arm}({arm*100//total}%)  "
                f"local_feat={lf}({lf*100//total}%)")

    for key in ["all", "sft_img", "rl_img", "sft_pc", "rl_pc"]:
        print(f"  [{key}]  {_pct_row(agg[key])}")

    for combo, cs in combo_stats.items():
        print(f"\n  {combo} (n={cs['n']}):")
        print(f"    aspect_ratio={cs['aspect_ratio_pct']}%  "
              f"mild={cs['aspect_ratio_mild_pct']}%  "
              f"local_feat={cs['local_feat_pct']}%")
        print(f"    asp_diff median={cs['asp_diff_median']:.3f}  p90={cs['asp_diff_p90']:.3f}  "
              f"vol_ratio median={cs['vol_ratio_median']:.3f}  IoU median={cs['iou_median']:.3f}")

    # ── write markdown report ──────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 0.5 — dim_error Sub-Classification",
        "**Date:** 2026-03-21",
        "**Script:** `tools/analyze_dim_errors.py`",
        "**Method:** Load pred + GT STL, normalise both to [-1,1]³, compare sorted",
        "normalised extents to classify geometric error type.",
        "",
        "Sub-types:",
        "- `aspect_ratio` (asp_diff ≥ 0.25): wrong proportions — one or more axes scaled differently",
        "- `aspect_ratio_mild` (0.12 ≤ asp_diff < 0.25): mild proportion mismatch",
        "- `local_feat` (asp_diff < 0.12): bounding box matches, IoU low due to holes/cutouts/features",
        "",
        "---",
        "",
        "## Taxonomy breakdown (all low-IoU success cases, IoU ∈ (0, 0.70])",
        "",
    ]

    # per-combo taxonomy table
    all_cats = ["dim_error", "wrong_primitive", "degenerate", "wrong_plane",
                "partial_geom", "feature_count", "no_code"]
    header = "| Combo | " + " | ".join(all_cats) + " | Total |"
    sep    = "|---" * (len(all_cats) + 2) + "|"
    lines += [header, sep]
    for combo, dataset, mesh_dir, model in combos:
        if combo not in per_combo_taxonomy:
            continue
        tc = per_combo_taxonomy[combo]
        total = sum(tc.values())
        cells = [str(tc.get(c, 0)) for c in all_cats]
        lines.append(f"| {combo} | " + " | ".join(cells) + f" | {total} |")
    lines += [""]

    # aggregated
    lines += [
        "---",
        "",
        "## dim_error sub-type breakdown",
        "",
        "### Aggregated (all combos combined)",
        "",
    ]
    all_total = sum(agg["all"].values())
    lines += [
        "| Sub-type | Count | % |",
        "|---|---:|---:|",
        f"| `aspect_ratio` (strong) | {agg['all'].get('aspect_ratio',0)} | "
        f"{agg['all'].get('aspect_ratio',0)*100//max(all_total,1)}% |",
        f"| `aspect_ratio_mild`     | {agg['all'].get('aspect_ratio_mild',0)} | "
        f"{agg['all'].get('aspect_ratio_mild',0)*100//max(all_total,1)}% |",
        f"| `local_feat`            | {agg['all'].get('local_feat',0)} | "
        f"{agg['all'].get('local_feat',0)*100//max(all_total,1)}% |",
        f"| **Total**               | **{all_total}** | 100% |",
        "",
    ]

    lines += [
        "### Per-combo breakdown",
        "",
        "| Combo | n_dim | aspect_ratio | aspect_mild | local_feat | "
        "asp_diff_med | asp_diff_p90 | vol_ratio_med | iou_med |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for combo in [c[0] for c in combos if c[0] in combo_stats]:
        cs = combo_stats[combo]
        dc = per_combo_dim.get(combo, {})
        n = cs["n"]
        ar  = dc.get("aspect_ratio", 0)
        arm = dc.get("aspect_ratio_mild", 0)
        lf  = dc.get("local_feat", 0)
        lines.append(
            f"| {combo} | {n} | {ar} ({cs['aspect_ratio_pct']}%) | "
            f"{arm} ({cs['aspect_ratio_mild_pct']}%) | "
            f"{lf} ({cs['local_feat_pct']}%) | "
            f"{cs['asp_diff_median']:.3f} | {cs['asp_diff_p90']:.3f} | "
            f"{cs['vol_ratio_median']:.3f} | {cs['iou_median']:.3f} |"
        )
    lines += [""]

    # volume bias table
    lines += [
        "---",
        "",
        "## Volume bias (pred vs GT, normalised space)",
        "",
        "| Combo | over>1.1 | near±10% | under<0.9 | median ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for combo in [c[0] for c in combos if c[0] in vol_bias]:
        vb = vol_bias[combo]
        lines.append(
            f"| {combo} | {vb['over_pct']}% | {vb['near_pct']}% | "
            f"{vb['under_pct']}% | {vb['median']:.3f} |"
        )
    lines += [""]

    # SFT vs RL comparison
    lines += [
        "---",
        "",
        "## SFT vs RL dim_error sub-type comparison",
        "",
        "| Group | n | aspect_ratio | local_feat |",
        "|---|---:|---:|---:|",
    ]
    for key in ["sft_img", "rl_img", "sft_pc", "rl_pc"]:
        d = agg[key]
        n = sum(d.values())
        if n == 0:
            continue
        ar = d.get("aspect_ratio", 0) + d.get("aspect_ratio_mild", 0)
        lf = d.get("local_feat", 0)
        lines.append(
            f"| {key} | {n} | {ar} ({ar*100//n}%) | {lf} ({lf*100//n}%) |"
        )
    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "*(auto-generated placeholder — fill in after reviewing numbers)*",
        "",
        "- **aspect_ratio dominant** → model generates wrong shape proportions → fix: improve",
        "  sketch coordinate grounding or add normalisation-aware training signal",
        "- **local_feat dominant** → model gets overall shape right but misses holes/features →",
        "  fix: add Chamfer distance precision reward (Option A in plan.md)",
        "- **Mixed** → both problems present; start with CD reward (broadly useful)",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
