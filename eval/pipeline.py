"""Inference + scoring pipeline for Cadrille eval.

Core design (same as tools/analyze_errors.py, generalised):
  - Background thread pool: prepares examples (renders img / samples pc)
  - Bounded queue: decouples prep from GPU batch inference
  - Main thread: drains queue → batched model.generate()
  - ThreadPoolExecutor: scores completed batches in subprocess workers
  - Results written to metadata.jsonl as they arrive (resume-safe)

Entry point:
    run_combo(model, processor, cfg, ckpt_label, ds_name, ds_cfg, modality, out_dir)

Gotchas:
  - collate() must be called with eval=True (not is_train) for inference;
    eval=False omits add_generation_prompt and includes the answer label (train mode).
  - eos_token_id must be model.config.eos_token_id (<|im_end|>, 151645), NOT
    model.config.video_token_id (151656). Using the video token as EOS causes the
    model to never stop, producing repetitive garbled output and 0% IoU on all cases.
"""
from __future__ import annotations

import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from common.meshio import render_img

_N_POINTS = 256

_SCORE_SCRIPT = textwrap.dedent(
    "import sys, json, tempfile, os, time\n"
    "import cadquery as cq\n"
    "import trimesh\n"
    "import numpy as np\n"
    "\n"
    "code        = sys.argv[1]\n"
    "gt_mesh_path= sys.argv[2]\n"
    "timeout     = float(sys.argv[3])\n"
    "\n"
    "def _iou(m1, m2):\n"
    "    import subprocess, tempfile, os\n"
    "    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f1:\n"
    "        f1.write(m1.export('stl')); p1=f1.name\n"
    "    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f2:\n"
    "        f2.write(m2.export('stl')); p2=f2.name\n"
    "    try:\n"
    "        r = subprocess.run(\n"
    "            ['python3', '-c',\n"
    "             f'import trimesh,numpy as np; '\n"
    "             f'a=trimesh.load({p1!r}); b=trimesh.load({p2!r}); '\n"
    "             f'a=trimesh.creation.box(a.bounding_box.extents) if not isinstance(a,trimesh.Trimesh) else a; '\n"
    "             f'b=trimesh.creation.box(b.bounding_box.extents) if not isinstance(b,trimesh.Trimesh) else b; '\n"
    "             f'bounds=np.stack([a.bounds,b.bounds]); mn=bounds.min(0)[0]; mx=bounds.max(0)[1]; '\n"
    "             f'pts=np.random.RandomState(0).uniform(mn,mx,(50000,3)); '\n"
    "             f'ia=a.contains(pts); ib=b.contains(pts); '\n"
    "             f'iou=float((ia&ib).sum()/(ia|ib).sum()) if (ia|ib).sum()>0 else 0.0; '\n"
    "             f'print(iou)'],\n"
    "            capture_output=True, text=True, timeout=timeout)\n"
    "        if r.returncode==0:\n"
    "            return float(r.stdout.strip())\n"
    "    finally:\n"
    "        os.unlink(p1); os.unlink(p2)\n"
    "    return None\n"
    "\n"
    "try:\n"
    "    ns = {}\n"
    "    exec(compile(code,'<gen>','exec'), ns)\n"
    "    result = ns.get('result') or ns.get('r') or [v for v in ns.values()\n"
    "             if hasattr(v,'val') and callable(getattr(v,'val',None))]\n"
    "    if isinstance(result, list): result = result[-1] if result else None\n"
    "    if result is None: raise ValueError('no result object')\n"
    "    mesh = trimesh.Trimesh(**trimesh.exchange.stl.load_stl(\n"
    "        io.BytesIO(result.val().toCompound().exportBrep() if False\n"
    "                   else result.val().exportStl().encode()\n"
    "                   if hasattr(result.val(),'exportStl')\n"
    "                   else b'')))\n"
    "except Exception as e:\n"
    "    print(json.dumps({'error_type':'runtime_error','error_msg':str(e),'iou':None}))\n"
    "    sys.exit(0)\n"
)


def _score_case(code: str, gt_mesh_path: str, timeout: float = 32.0) -> dict:
    """Score a generated code string. Returns dict with iou, error_type, etc."""
    _SCORE_PY = Path(__file__).parent.parent / 'rl' / 'reward.py'
    # unused — kept for reference

    try:
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    f"\nimport sys, json\n"
                    f"sys.path.insert(0, '{Path(__file__).parent.parent}')\n"
                    f"from common.metrics import compute_metrics\n"
                    f"code = {repr(code)}\n"
                    f"iou_reward, cd = compute_metrics(code, {repr(gt_mesh_path)}, timeout={timeout})\n"
                    f"if iou_reward <= -2.0:\n"
                    f"    print(json.dumps({{'error_type':'syntax_error','iou':None,'cd':None}}))\n"
                    f"elif iou_reward <= -1.0:\n"
                    f"    print(json.dumps({{'error_type':'runtime_error','iou':None,'cd':None}}))\n"
                    f"elif iou_reward == 0.0:\n"
                    f"    print(json.dumps({{'error_type':'zero_iou','iou':0.0,'cd':cd}}))\n"
                    f"else:\n"
                    f"    print(json.dumps({{'error_type':'success','iou':iou_reward,'cd':cd}}))\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
    except subprocess.TimeoutExpired:
        return {'error_type': 'timeout', 'iou': None, 'cd': None}
    except Exception:
        return {'error_type': 'runtime_error', 'iou': None, 'cd': None}

    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout.strip().splitlines()[-1])

    return {'error_type': 'runtime_error', 'iou': None, 'cd': None}


def _exec_and_export_stl(code: str, timeout: float = 32.0) -> Optional[bytes]:
    # unused in direct path but kept for save_stl support
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                f"\nimport sys\n"
                f"sys.path.insert(0, '{Path(__file__).parent.parent}')\n"
                f"code = {repr(code)}\n"
                f"ns = {{}}\n"
                f"exec(compile(code, '<gen>', 'exec'), ns)\n"
                f"result = ns.get('result') or ns.get('r')\n"
                f"if result is None:\n"
                f"    candidates = [v for v in ns.values()\n"
                f"                  if hasattr(v, 'val') and callable(getattr(v, 'val', None))]\n"
                f"    result = candidates[-1] if candidates else None\n"
                f"if result is not None:\n"
                f"    import sys\n"
                f"    sys.stdout.buffer.write(result.val().exportStl().encode()\n"
                f"                            if hasattr(result.val(), 'exportStl')\n"
                f"                            else b'')\n"
            ),
        ],
        capture_output=True,
        timeout=timeout + 5,
    )
    if result.returncode == 0 and result.stdout:
        return result.stdout
    try:
        pass
    except Exception:
        pass
    return None


def _load_all_stls(ds_path: Path, n_samples: Optional[int] = None) -> list[str]:
    stls = sorted(str(p) for p in ds_path.glob('*.stl'))
    if n_samples is not None:
        stls = stls[:n_samples]
    return stls


_DESCRIPTION = 'Generate cadquery code'


def _prep_img(gt_mesh_path: str) -> Optional[dict]:
    try:
        stem = Path(gt_mesh_path).stem
        # render_img handles prerendered PNG fallback automatically
        img_dict = render_img(gt_mesh_path)
        return {
            'stem': stem,
            'gt_mesh_path': gt_mesh_path,
            'description': _DESCRIPTION,
            'file_name': stem,
            **img_dict,
        }
    except Exception:
        return None


def _prep_pc(gt_mesh_path: str) -> Optional[dict]:
    from dataset import mesh_to_point_cloud

    try:
        stem = Path(gt_mesh_path).stem
        # load mesh and sample point cloud
        mesh = trimesh.load(gt_mesh_path)
        pc = mesh_to_point_cloud(mesh, _N_POINTS)
        pc = (pc - 0.5) * 2
        return {
            'stem': stem,
            'gt_mesh_path': gt_mesh_path,
            'description': _DESCRIPTION,
            'file_name': stem,
            'point_cloud': pc,
        }
    except Exception:
        return None


def _prep_worker(
    stl_paths: list[str],
    modality: str,
    out_q: queue.Queue,
    done_event: threading.Event,
    n_threads: int,
) -> None:
    def _prep(path):
        if modality == 'img':
            return _prep_img(path)
        else:
            return _prep_pc(path)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_prep, p): p for p in stl_paths}
        for fut in as_completed(futures):
            item = fut.result()
            if item is not None:
                out_q.put(item)

    done_event.set()


def run_combo(
    stl_paths: list[str],
    modality: str,
    out_dir: Path,
    batch_size: int = 8,
    max_new_tokens: int = 768,
    score_workers: int = 4,
    prep_threads: int = 2,
    queue_size: int = 32,
    save_code: bool = True,
    save_stl: bool = True,
    # model and processor are passed as keyword args from runner
    model=None,
    processor=None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'metadata.jsonl'

    done_ids = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r['case_id'])

    todo = [p for p in stl_paths if Path(p).stem not in done_ids]

    if not todo:
        print(f'  All {len(done_ids)} cases already done. Skipping.')
        return _summarize_metadata(meta_path)

    print(f'  {len(todo)} cases to run (already done: {len(done_ids)})', flush=True)

    prep_q = queue.Queue(maxsize=queue_size)
    done_event = threading.Event()

    prep_thread = threading.Thread(
        target=_prep_worker,
        args=(todo, modality, prep_q, done_event, prep_threads),
        daemon=True,
    )
    prep_thread.start()

    device = next(model.parameters()).device
    meta_file = open(meta_path, 'a')
    score_pool = ThreadPoolExecutor(max_workers=score_workers)
    pending = []

    def _flush_pending(wait_all: bool = False) -> None:
        remaining = []
        check = list(pending)
        for fut, stem, gt_mesh_path, code in check:
            if not wait_all and not fut.done():
                remaining.append((fut, stem, gt_mesh_path, code))
                continue

            score = fut.result()
            iou = score.get('iou')
            cd = score.get('cd')
            etype = score.get('error_type', 'runtime_error')

            rec = {
                'case_id':    stem,
                'iou':        iou,
                'error_type': etype,
                'error_msg':  score.get('error_msg'),
                'cd':         cd,
                'code_len':   len(code),
            }

            meta_file.write(json.dumps(rec) + '\n')
            meta_file.flush()

            if save_code:
                (out_dir / f'{stem}.py').write_text(code)

            if save_stl and etype == 'success':
                stl_bytes = _exec_and_export_stl(code)
                if stl_bytes:
                    (out_dir / f'{stem}.stl').write_bytes(stl_bytes)

        pending.clear()
        pending.extend(remaining)

    batch = []
    pbar = tqdm(len(todo), desc=modality, leave=False)

    def _drain_batch() -> None:
        if not batch:
            return

        to_infer = []
        cached = []

        for item in batch:
            code_cache = out_dir / f'{item["stem"]}.py'
            if code_cache.exists():
                cached.append((item, code_cache.read_text()))
            else:
                to_infer.append(item)

        generated = {}

        if to_infer:
            if hasattr(model, 'rope_deltas'):
                model.rope_deltas = None

            collate_items = [
                {k: v for k, v in it.items() if k not in ('stem', 'gt_mesh_path')}
                for it in to_infer
            ]

            b = collate(collate_items, processor, _N_POINTS, eval=True)

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=b['input_ids'].to(device),
                    attention_mask=b['attention_mask'].to(device),
                    point_clouds=b['point_clouds'].to(device),
                    is_pc=b['is_pc'].to(device),
                    is_img=b['is_img'].to(device),
                    pixel_values_videos=b.get('pixel_values_videos').to(device) if b.get('pixel_values_videos') is not None else None,
                    video_grid_thw=b.get('video_grid_thw').to(device) if b.get('video_grid_thw') is not None else None,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=model.config.eos_token_id,
                )

            prompt_len = b['input_ids'].shape[1]
            for j, item in enumerate(to_infer):
                code = processor.decode(out_ids[j, prompt_len:None], skip_special_tokens=True)
                generated[item['stem']] = code

        for item, code in cached:
            fut = score_pool.submit(_score_case, code, item['gt_mesh_path'])
            pending.append((fut, item['stem'], item['gt_mesh_path'], code))

        for item in to_infer:
            code = generated[item['stem']]
            fut = score_pool.submit(_score_case, code, item['gt_mesh_path'])
            pending.append((fut, item['stem'], item['gt_mesh_path'], code))

        pbar.update(len(batch))
        batch.clear()
        _flush_pending(wait_all=False)

    # Main drain loop
    while True:
        if done_event.is_set() and prep_q.empty():
            break
        try:
            item = prep_q.get(timeout=0.1)
        except queue.Empty:
            if done_event.is_set():
                break
            continue
        batch.append(item)
        if len(batch) >= batch_size:
            _drain_batch()

    if batch:
        _drain_batch()

    _flush_pending(wait_all=True)
    pbar.close()
    meta_file.close()
    score_pool.shutdown(wait=True)

    return _summarize_metadata(meta_path)


def _summarize_metadata(meta_path: Path) -> dict:
    records = []
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                records.append(json.loads(line))

    if not records:
        return {'n': 0}

    ious = [r['iou'] for r in records if r.get('iou') is not None]
    cds = [r['cd'] for r in records if r.get('cd') is not None]
    fail = sum(1 for r in records if r['error_type'] != 'success')

    err_types = {}
    for r in records:
        err_types[r['error_type']] = err_types.get(r['error_type'], 0) + 1

    return {
        'n':            len(records),
        'n_success':    len(records) - fail,
        'failure_rate': fail / len(records),
        'mean_iou':     float(np.mean(ious)) if ious else 0.0,
        'median_iou':   float(np.median(ious)) if ious else 0.0,
        'mean_cd':      float(np.mean(cds)) if cds else None,
        'error_types':  err_types,
    }
