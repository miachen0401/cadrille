"""Build a curated wandb workspace for cadrille SFT runs.

Out-of-the-box wandb shows every logged metric as its own panel
(100+ panels for our run with 5-bucket eval). This script creates a
saved workspace view with the metrics worth looking at, organised
into sections:

    Training                train/eval loss overlay, grad norm, lr
    IoU & exec              greedy IoU + exec_rate per IoU bucket
    Sampling                max_iou@8 + pass>0.5 per IoU bucket
    Ops loss                op_loss_cos_weighted per ops bucket
    Recall (per bucket)     op_macro_recall + rare_op_macro_recall
    Diversity               distinct_ops + distinct_codes_frac
    text2cad                text2cad recall + rare_recall + op_loss
    Curriculum              eval/global_step (x-axis sanity)

Run once per project:

    uv run python -m scripts.analysis.setup_wandb_dashboard \
        --entity hula-the-cat --project cadrille-sft \
        --name cadrille-default

Or via positional:

    uv run python -m scripts.analysis.setup_wandb_dashboard \
        hula-the-cat/cadrille-sft

Idempotent — re-running overwrites the named workspace.
"""
from __future__ import annotations

import argparse
import os
import sys

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr


# ---------------------------------------------------------------------------
# Metric key catalog (matches train/sft/online_eval.py + HF Trainer auto)
# ---------------------------------------------------------------------------
IOU_BUCKETS = ['BenchCAD val', 'DeepCAD test', 'Fusion360 test']
OPS_BUCKETS = ['BenchCAD val', 'recode20k train', 'text2cad train']
ALL_BUCKETS = ['BenchCAD val', 'recode20k train', 'text2cad train',
               'DeepCAD test', 'Fusion360 test']


def _img(bucket: str, leaf: str) -> str:
    """eval/img/{bucket}/{leaf}"""
    return f'eval/img/{bucket}/{leaf}'


def _text(bucket: str, leaf: str) -> str:
    """eval/text/{bucket}/{leaf}"""
    return f'eval/text/{bucket}/{leaf}'


def _bucket_keys(buckets: list[str], leaf: str, modality: str = 'img') -> list[str]:
    return [f'eval/{modality}/{b}/{leaf}' for b in buckets]


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def _line(title: str, y_keys: list[str], smoothing: float = 0.0) -> wr.LinePlot:
    return wr.LinePlot(
        title=title,
        y=y_keys,
        smoothing_factor=smoothing,
    )


def build_workspace(entity: str, project: str, name: str) -> ws.Workspace:
    sections = [
        ws.Section(
            name='Training',
            panels=[
                _line('train/loss + eval/loss', ['train/loss', 'eval/loss'],
                      smoothing=0.5),
                _line('grad_norm', ['train/grad_norm']),
                _line('learning_rate', ['train/learning_rate']),
            ],
        ),
        ws.Section(
            name='IoU & exec_rate (greedy, per IoU bucket)',
            panels=[
                _line('IoU mean', _bucket_keys(IOU_BUCKETS, 'IoU mean')),
                _line('exec_rate', _bucket_keys(IOU_BUCKETS, 'exec_rate')),
            ],
        ),
        ws.Section(
            name='max_iou@8 (t=1) — sampling head-room',
            panels=[
                _line('max_iou@8 (t=1.0)',
                      _bucket_keys(IOU_BUCKETS, 'max_iou@8 (t=1.0)')),
                _line('pass_iou_0.5@8 (t=1.0)',
                      _bucket_keys(IOU_BUCKETS, 'pass_iou_0.5@8 (t=1.0)')),
            ],
        ),
        ws.Section(
            name='Ops loss (weighted cosine)',
            panels=[
                _line('op_loss_cos_weighted',
                      [_img(b, 'op_loss_cos_weighted') for b in
                       ('BenchCAD val', 'recode20k train')]
                      + [_text('text2cad train', 'op_loss_cos_weighted')]),
            ],
        ),
        ws.Section(
            name='Recall — overall vs rare-op cohort',
            panels=[
                _line('op_macro_recall',
                      [_img(b, 'op_macro_recall') for b in
                       ('BenchCAD val', 'recode20k train')]
                      + [_text('text2cad train', 'op_macro_recall')]),
                _line('rare_op_macro_recall  ← mode-collapse signal',
                      [_img(b, 'rare_op_macro_recall') for b in
                       ('BenchCAD val', 'recode20k train')]
                      + [_text('text2cad train', 'rare_op_macro_recall')]),
                _line('ops_zero_recall_count',
                      [_img(b, 'ops_zero_recall_count') for b in
                       ('BenchCAD val', 'recode20k train')]
                      + [_text('text2cad train', 'ops_zero_recall_count')]),
            ],
        ),
        ws.Section(
            name='Diversity (sampling collapse early-warning)',
            panels=[
                _line('distinct_ops (out of 30)',
                      [_img(b, 'distinct_ops') for b in
                       ('BenchCAD val', 'recode20k train',
                        'DeepCAD test', 'Fusion360 test')]
                      + [_text('text2cad train', 'distinct_ops')]),
                _line('distinct_codes_frac',
                      [_img(b, 'distinct_codes_frac') for b in
                       ('BenchCAD val', 'recode20k train',
                        'DeepCAD test', 'Fusion360 test')]
                      + [_text('text2cad train', 'distinct_codes_frac')]),
                _line('pred_count_mean',
                      [_img(b, 'pred_count_mean') for b in
                       ('BenchCAD val', 'recode20k train')]
                      + [_text('text2cad train', 'pred_count_mean')]),
            ],
        ),
        ws.Section(
            name='Per-op recall — chamfer/revolve/sphere/fillet',
            panels=[
                _line(f'{op}',
                      [_img(b, f'op_recall/{op}') for b in
                       ('BenchCAD val', 'recode20k train')]
                      + [_text('text2cad train', f'op_recall/{op}')])
                for op in ('chamfer', 'revolve', 'sphere', 'fillet',
                           'hole', 'cut')
            ],
        ),
    ]

    return ws.Workspace(
        name=name,
        entity=entity,
        project=project,
        sections=sections,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('project_path', nargs='?', default=None,
                   help='entity/project (alternative to --entity / --project)')
    p.add_argument('--entity', default=None, help='wandb entity (org / user)')
    p.add_argument('--project', default=None, help='wandb project name')
    p.add_argument('--name', default='cadrille-default',
                   help='workspace name (default: cadrille-default)')
    args = p.parse_args()

    if args.project_path:
        if '/' not in args.project_path:
            print(f'project_path must be entity/project, got {args.project_path!r}',
                  file=sys.stderr)
            sys.exit(2)
        args.entity, args.project = args.project_path.split('/', 1)

    if not (args.entity and args.project):
        print('need --entity + --project (or positional entity/project)',
              file=sys.stderr)
        sys.exit(2)

    if not os.environ.get('WANDB_API_KEY'):
        print('WANDB_API_KEY env var not set; cannot authenticate.',
              file=sys.stderr)
        sys.exit(2)

    print(f'Building workspace {args.name!r} on {args.entity}/{args.project} ...')
    workspace = build_workspace(args.entity, args.project, args.name)
    saved = workspace.save()
    print(f'  ✅ saved → {saved.url}')


if __name__ == '__main__':
    main()
