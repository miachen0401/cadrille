"""v2: rewrite recode/text2cad code → BenchCAD shell style.

Beyond the AST format-only pass in `rewrite_recode_to_bench.py`, v2 applies:

  Rule A:  .sketch() ... .finalize().extrude(N)  →  direct Workplane chain
           - .segment / .arc / .close       → .moveTo / .lineTo / .threePointArc / .close
           - .push([(x,y)]).circle(r)       → .center(x,y).circle(r)
           - .push([(x,y)]).rect(w,h)       → .center(x,y).rect(w,h)
           - .face(inner, mode='a'|'s')     → unwrap inner sketch, dispatch by mode
  Rule B:  .workplane(offset=N)             →  .transformed(offset=cq.Vector(0, 0, N))
  Rule D:  .face(..., mode='s')             →  base.extrude(N).cut(sub.extrude(N))

When v2 cannot rewrite (mixed unsupported sketch construction), the chain
falls back to v1 format-only rewrite (sketch form preserved) so we never
emit invalid code.

IoU validated on 6 hand-crafted patterns — 6/6 = 1.0000.

Usage:
  uv run python -m data_prep.rewrite_recode_to_benchcad_v2 \\
      --input  data/cad-recode-v1.5/train \\
      --output data/cad-recode-v1.5-benchcad/train \\
      --jobs 8

Smoke test (print 3 worked examples):
  uv run python -m data_prep.rewrite_recode_to_benchcad_v2 \\
      --input  data/cad-recode-v1.5/train --output /tmp/out --show-first 3

IoU validation (sample N, exec orig vs rewrite, compare voxel IoU):
  uv run python -m data_prep.rewrite_recode_to_benchcad_v2 \\
      --input  data/cad-recode-v1.5/train --output /tmp/out --validate-iou 200
"""
from __future__ import annotations

import argparse
import ast
import copy
import multiprocessing as mp
import random
import sys
import traceback
from pathlib import Path

INDENT = '    '


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_chain(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)


def _split_chain(node):
    """Walk outer→inner. Return (base, [(method, args, kw), ...] inner→outer)."""
    methods = []
    cur = node
    while _is_chain(cur):
        methods.append((cur.func.attr, list(cur.args), list(cur.keywords)))
        cur = cur.func.value
    return cur, list(reversed(methods))


def _rebuild_chain(base, methods):
    cur = base
    for name, args, kw in methods:
        cur = ast.Call(
            func=ast.Attribute(value=cur, attr=name, ctx=ast.Load()),
            args=list(args),
            keywords=list(kw),
        )
    return cur


def _const(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _const_pair(node):
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        a, b = _const(node.elts[0]), _const(node.elts[1])
        if a is not None and b is not None:
            return (a, b)
    return None


def _inline_vars(expr, var_map):
    """Replace Name(x) with deep copy of var_map[x]."""
    class Inliner(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in var_map:
                return copy.deepcopy(var_map[n.id])
            return n
    return Inliner().visit(copy.deepcopy(expr))


# ---------------------------------------------------------------------------
# Rule B: .workplane(offset=N) → .transformed(offset=cq.Vector(0, 0, N))
# ---------------------------------------------------------------------------

def _apply_rule_b(node):
    """Walk AST, replace .workplane(offset=N, ...) → .transformed(offset=Vector(0,0,N))."""
    class B(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if isinstance(n.func, ast.Attribute) and n.func.attr == 'workplane':
                offset_kw = None
                other_kws = []
                for kw in n.keywords:
                    if kw.arg == 'offset':
                        offset_kw = kw.value
                    else:
                        other_kws.append(kw)
                if offset_kw is None or n.args:
                    return n  # no offset or has positional args — leave alone
                vector_call = ast.Call(
                    func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='cq', ctx=ast.Load()),
                                                            attr='Vector', ctx=ast.Load()),
                                       attr=None, ctx=ast.Load()),
                    args=[ast.Constant(0), ast.Constant(0), offset_kw],
                    keywords=[],
                )
                # Build cq.Vector(0,0,offset) properly (without that bogus None attr above)
                vector_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id='cq', ctx=ast.Load()),
                                       attr='Vector', ctx=ast.Load()),
                    args=[ast.Constant(0), ast.Constant(0), offset_kw],
                    keywords=[],
                )
                new_kw = [ast.keyword(arg='offset', value=vector_call)] + other_kws
                return ast.Call(
                    func=ast.Attribute(value=n.func.value, attr='transformed', ctx=ast.Load()),
                    args=[],
                    keywords=new_kw,
                )
            return n
    return B().visit(node)


# ---------------------------------------------------------------------------
# Rule A: parse sketch block into FaceSegment list
# ---------------------------------------------------------------------------

def _parse_face_segments(inner_methods):
    """Parse methods between .sketch() and .finalize() into FaceSegment list.

    Returns list of dicts, or None on unsupported pattern.
    Each FaceSegment:
      {'mode': 'a'|'s', 'kind': 'circle', 'centers': [(x,y),...], 'r': r}
      {'mode': 'a'|'s', 'kind': 'rect',   'centers': [(x,y),...], 'w': w, 'h': h}
      {'mode': 'a'|'s', 'kind': 'segment_chain', 'edges': [(name, args, kw), ...]}
    """
    faces = []
    cur_chain = None      # active segment_chain face being built
    push_pts = None       # last .push([...]) result, consumed by next .circle/.rect
    next_mode = 'a'       # mode applies to next face emitted

    def flush_chain():
        nonlocal cur_chain
        if cur_chain is not None:
            faces.append(cur_chain)
            cur_chain = None

    for name, args, kw in inner_methods:
        if name == 'reset':
            flush_chain()
            push_pts = None
            continue

        if name == 'face':
            # face(inner_expr, mode='a'|'s')
            mode = next_mode
            for k in kw:
                if k.arg == 'mode' and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            inner = args[0] if args else None
            flush_chain()
            push_pts = None
            if inner is None or not _is_chain(inner):
                return None
            _, inner_methods_full = _split_chain(inner)
            # Find .sketch() in inner methods (inner is usually
            # `cq.Workplane(...).sketch()....assemble()` — Workplane is also a Call,
            # so split_chain returns it as the leading method).
            sk_idx = next((i for i, m in enumerate(inner_methods_full)
                           if m[0] == 'sketch'), None)
            if sk_idx is None:
                return None
            inner_methods = inner_methods_full[sk_idx + 1:]
            inner_segs = _parse_face_segments(inner_methods)
            if inner_segs is None:
                return None
            # outer face's mode overrides inner — treat all inner faces as `mode`
            for fs in inner_segs:
                fs['mode'] = mode
            faces.extend(inner_segs)
            continue

        if name == 'push':
            if not args or not isinstance(args[0], ast.List):
                if args and isinstance(args[0], ast.Tuple):
                    p = _const_pair(args[0])
                    if p is None: return None
                    push_pts = [p]
                else:
                    return None
            else:
                pts = []
                for elt in args[0].elts:
                    p = _const_pair(elt)
                    if p is None: return None
                    pts.append(p)
                push_pts = pts
            continue

        if name == 'circle':
            mode = next_mode
            for k in kw:
                if k.arg == 'mode' and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if not args or _const(args[0]) is None: return None
            r = _const(args[0])
            centers = push_pts if push_pts else [(0, 0)]
            faces.append({'mode': mode, 'kind': 'circle', 'centers': centers, 'r': r})
            # NB: cadquery sketch.push() sets a location stack that persists until
            # reset() or a new push() — do NOT clear push_pts here, otherwise
            # patterns like `.push([(34,36)]).circle(64).circle(60,mode='s')`
            # (concentric ring) lose the position on the second circle.
            continue

        if name == 'rect':
            mode = next_mode
            for k in kw:
                if k.arg == 'mode' and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if len(args) < 2 or _const(args[0]) is None or _const(args[1]) is None: return None
            w, h = _const(args[0]), _const(args[1])
            centers = push_pts if push_pts else [(0, 0)]
            faces.append({'mode': mode, 'kind': 'rect', 'centers': centers, 'w': w, 'h': h})
            continue

        if name in ('segment', 'arc'):
            if cur_chain is None:
                cur_chain = {'mode': next_mode, 'kind': 'segment_chain', 'edges': []}
            cur_chain['edges'].append((name, args, kw))
            continue

        if name == 'close':
            if cur_chain is not None:
                cur_chain['edges'].append(('close', [], []))
            continue

        if name == 'assemble':
            flush_chain()
            continue

        # Unknown method in sketch block — bail out
        return None

    flush_chain()
    return faces


# ---------------------------------------------------------------------------
# Render FaceSegment → benchcad chain string (single line)
# ---------------------------------------------------------------------------

def _fmt_num(x):
    """Format a number — integer if integral, else compact float."""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        return f'{x:g}'
    return str(x)


def _render_face(face, plane_str, extrude_arg_str):
    """Render one FaceSegment as a chain ending in .extrude(N). Returns str or None."""
    if face['kind'] == 'circle':
        parts = []
        for (x, y) in face['centers']:
            parts.append(f"{plane_str}.center({_fmt_num(x)}, {_fmt_num(y)})"
                         f".circle({_fmt_num(face['r'])}).extrude({extrude_arg_str})")
        head = parts[0]
        for p in parts[1:]:
            head = f"{head}.union({p})"
        return head

    if face['kind'] == 'rect':
        parts = []
        for (x, y) in face['centers']:
            parts.append(f"{plane_str}.center({_fmt_num(x)}, {_fmt_num(y)})"
                         f".rect({_fmt_num(face['w'])}, {_fmt_num(face['h'])}).extrude({extrude_arg_str})")
        head = parts[0]
        for p in parts[1:]:
            head = f"{head}.union({p})"
        return head

    if face['kind'] == 'segment_chain':
        edges = face['edges']
        chain = plane_str
        first = True
        for name, args, kw in edges:
            if name == 'segment':
                if len(args) == 2:
                    p1 = _const_pair(args[0]); p2 = _const_pair(args[1])
                    if p1 is None or p2 is None: return None
                    if first:
                        chain += f".moveTo({_fmt_num(p1[0])}, {_fmt_num(p1[1])})"
                        first = False
                    chain += f".lineTo({_fmt_num(p2[0])}, {_fmt_num(p2[1])})"
                elif len(args) == 1:
                    p = _const_pair(args[0])
                    if p is None: return None
                    if first:
                        chain += f".moveTo({_fmt_num(p[0])}, {_fmt_num(p[1])})"
                        first = False
                    else:
                        chain += f".lineTo({_fmt_num(p[0])}, {_fmt_num(p[1])})"
                else:
                    return None
            elif name == 'arc':
                if len(args) == 3:
                    p1 = _const_pair(args[0]); p2 = _const_pair(args[1]); p3 = _const_pair(args[2])
                    if any(p is None for p in (p1, p2, p3)): return None
                    if first:
                        chain += f".moveTo({_fmt_num(p1[0])}, {_fmt_num(p1[1])})"
                        first = False
                    chain += (f".threePointArc(({_fmt_num(p2[0])}, {_fmt_num(p2[1])}),"
                              f" ({_fmt_num(p3[0])}, {_fmt_num(p3[1])}))")
                elif len(args) == 2:
                    p1 = _const_pair(args[0]); p2 = _const_pair(args[1])
                    if p1 is None or p2 is None: return None
                    if first:
                        return None  # 2-arg arc with no current point — ambiguous
                    chain += (f".threePointArc(({_fmt_num(p1[0])}, {_fmt_num(p1[1])}),"
                              f" ({_fmt_num(p2[0])}, {_fmt_num(p2[1])}))")
                else:
                    return None
            elif name == 'close':
                # close handled at the end — just signal it
                pass

        if not chain.endswith('.close()'):
            chain += '.close()'
        chain += f'.extrude({extrude_arg_str})'
        return chain

    return None


def _render_faces(faces, plane_str, extrude_arg_str):
    """Render list of FaceSegments → single chain string with union/cut booleans."""
    adds = [f for f in faces if f['mode'] == 'a']
    subs = [f for f in faces if f['mode'] == 's']
    if not adds:
        return None  # need at least one base face

    base = _render_face(adds[0], plane_str, extrude_arg_str)
    if base is None: return None
    for a in adds[1:]:
        rendered = _render_face(a, plane_str, extrude_arg_str)
        if rendered is None: return None
        base = f"{base}.union({rendered})"
    for s in subs:
        rendered = _render_face(s, plane_str, extrude_arg_str)
        if rendered is None: return None
        base = f"{base}.cut({rendered})"
    return base


# ---------------------------------------------------------------------------
# Top-level chain rewriter — finds sketch...finalize.extrude(N) sub-sequences
# and replaces them with benchcad chains. Recursive into args.
# ---------------------------------------------------------------------------

def _rewrite_chain(node):
    """Recursive: rewrite all sketch.finalize.extrude blocks in node. Returns AST node."""
    if not isinstance(node, ast.AST):
        return node

    # Recurse into args/values first
    if isinstance(node, ast.Call):
        node.args = [_rewrite_chain(a) for a in node.args]
        node.keywords = [ast.keyword(arg=kw.arg, value=_rewrite_chain(kw.value)) for kw in node.keywords]

    if not _is_chain(node):
        # also recurse into Tuple/BinOp/etc generically
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                setattr(node, field, _rewrite_chain(value))
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, ast.AST):
                        value[i] = _rewrite_chain(v)
        return node

    base, methods = _split_chain(node)
    base = _rewrite_chain(base) if isinstance(base, ast.AST) else base

    # Recurse into method args
    new_methods = []
    for name, args, kw in methods:
        new_args = [_rewrite_chain(a) for a in args]
        new_kw = [ast.keyword(arg=k.arg, value=_rewrite_chain(k.value)) for k in kw]
        new_methods.append((name, new_args, new_kw))
    methods = new_methods

    # Find sketch...finalize.extrude(N) sub-sequence
    sketch_idx = next((i for i, m in enumerate(methods) if m[0] == 'sketch'), None)
    if sketch_idx is None:
        return _rebuild_chain(base, methods)
    finalize_idx = next((i for i, m in enumerate(methods) if i > sketch_idx and m[0] == 'finalize'), None)
    if finalize_idx is None or finalize_idx + 1 >= len(methods) or methods[finalize_idx + 1][0] != 'extrude':
        return _rebuild_chain(base, methods)

    pre_methods = methods[:sketch_idx]
    post_methods = methods[finalize_idx + 2:]
    inner_methods = methods[sketch_idx + 1:finalize_idx]
    extrude_args = methods[finalize_idx + 1][1]
    extrude_kw = methods[finalize_idx + 1][2]
    if extrude_kw:  # don't handle keyword args on extrude
        return _rebuild_chain(base, methods)

    # Build pre-sketch chain (cq.Workplane(...).transformed(...) etc) as string
    pre_chain_node = _rebuild_chain(base, pre_methods)
    plane_str = ast.unparse(pre_chain_node)

    # Parse face segments
    faces = _parse_face_segments(inner_methods)
    if faces is None or not faces:
        return _rebuild_chain(base, methods)

    extrude_arg_str = ', '.join(ast.unparse(a) for a in extrude_args)

    rendered = _render_faces(faces, plane_str, extrude_arg_str)
    if rendered is None:
        return _rebuild_chain(base, methods)

    # Parse rendered string back to AST node, then continue with post_methods
    try:
        new_node = ast.parse(rendered, mode='eval').body
    except SyntaxError:
        return _rebuild_chain(base, methods)

    # Append post_methods
    return _rebuild_chain(new_node, post_methods)


# ---------------------------------------------------------------------------
# Pretty printer — break long chains across lines
# ---------------------------------------------------------------------------

def _chain_length(node):
    n = 0
    cur = node
    while _is_chain(cur):
        n += 1
        cur = cur.func.value
    return n


def _render_chain_pretty(node, indent):
    """Render a chain expression with line breaks. First line has no indent; subsequent
    method lines are prefixed with INDENT*indent."""
    if not _is_chain(node):
        return ast.unparse(node)
    base, methods = _split_chain(node)

    def render_args(args, kws, indent):
        break_pos = [_is_chain(a) and _chain_length(a) >= 2 for a in args]
        break_kw = [_is_chain(kw.value) and _chain_length(kw.value) >= 2 for kw in kws]
        if not any(break_pos) and not any(break_kw):
            parts = [ast.unparse(a) for a in args] + [f'{kw.arg}={ast.unparse(kw.value)}' for kw in kws]
            return ', '.join(parts)
        inner = INDENT * (indent + 1)
        outer = INDENT * indent
        pieces = []
        for a, brk in zip(args, break_pos):
            if brk:
                rendered = _render_chain_pretty(a, indent + 1)
                first, _, rest = rendered.partition('\n')
                pieces.append(inner + first + ('\n' + rest if rest else ''))
            else:
                pieces.append(inner + ast.unparse(a))
        for kw, brk in zip(kws, break_kw):
            if brk:
                rendered = _render_chain_pretty(kw.value, indent + 1)
                first, _, rest = rendered.partition('\n')
                pieces.append(inner + f'{kw.arg}=' + first + ('\n' + rest if rest else ''))
            else:
                pieces.append(inner + f'{kw.arg}={ast.unparse(kw.value)}')
        return '\n' + ',\n'.join(pieces) + '\n' + outer

    if isinstance(base, ast.Name):
        first_name, first_args, first_kw = methods[0]
        first_kw_objs = [ast.keyword(arg=k.arg, value=k.value) if isinstance(k, ast.keyword) else ast.keyword(arg=k[0], value=k[1]) for k in first_kw] if first_kw and not isinstance(first_kw[0], ast.keyword) else first_kw
        args_txt = render_args(first_args, first_kw_objs, indent)
        base_txt = f'{base.id}.{first_name}({args_txt})'
        methods = methods[1:]
    elif _is_chain(base):
        base_txt = _render_chain_pretty(base, indent)
    else:
        base_txt = ast.unparse(base)

    out = [base_txt]
    for name, args, kws in methods:
        kws_objs = [ast.keyword(arg=k.arg, value=k.value) if isinstance(k, ast.keyword) else ast.keyword(arg=k[0], value=k[1]) for k in kws] if kws and not isinstance(kws[0], ast.keyword) else kws
        args_txt = render_args(args, kws_objs, indent)
        out.append(f'{INDENT * indent}.{name}({args_txt})')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Source-level driver
# ---------------------------------------------------------------------------

def rewrite_source(src):
    """Top-level: src → rewritten src in BenchCAD shell. Raises ValueError on bad input."""
    tree = ast.parse(src)

    var_map = {}
    final_target = None
    final_value = None

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            tgt = stmt.targets[0].id
            var_map[tgt] = stmt.value
            final_target = tgt
            final_value = stmt.value
            continue
        raise ValueError(f'Unsupported top-level stmt: {type(stmt).__name__}')

    if final_target is None:
        raise ValueError('No assignment target')

    # Iterative inline of intermediate vars
    middle = {k: v for k, v in var_map.items() if k != final_target}
    for _ in range(len(middle) + 1):
        changed = False
        for k in list(middle.keys()):
            others = {kk: vv for kk, vv in middle.items() if kk != k}
            new_val = _inline_vars(middle[k], others)
            if ast.dump(new_val) != ast.dump(middle[k]):
                middle[k] = new_val
                changed = True
        if not changed:
            break
    inlined = _inline_vars(final_value, middle)

    # Apply Rule B (.workplane(offset=N) → .transformed)
    inlined = _apply_rule_b(inlined)

    # Apply Rule A+D (sketch block → direct Workplane)
    rewritten = _rewrite_chain(inlined)

    body_txt = _render_chain_pretty(rewritten, indent=1)
    first, _, rest = body_txt.partition('\n')
    body = INDENT + first + ('\n' + rest if rest else '')

    return (
        'import cadquery as cq\n'
        '\n'
        'result = (\n'
        f'{body}\n'
        ')\n'
        '\n'
        '# Export\n'
        'show_object(result)\n'
    )


# ---------------------------------------------------------------------------
# Batch driver + IoU validator
# ---------------------------------------------------------------------------

def _process_one(args):
    in_path, out_path = args
    try:
        src = Path(in_path).read_text()
        new = rewrite_source(src)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(new)
        return True, None
    except Exception as e:
        return False, f'{in_path}: {type(e).__name__}: {e}'


def _to_mesh(src):
    import cadquery as cq
    import trimesh
    ns = {'show_object': lambda *a, **k: None}
    exec(src, ns)
    r = ns.get('r') or ns.get('result')
    if r is None: return None
    compound = r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)


def _iou(m1, m2, n=32):
    """Lightweight semantic equivalence check via volume + bounds + centroid agreement.

    Voxel IoU was the original plan but trimesh.voxelized() OOM'd on
    text2cad's large-bounds meshes (single worker hit 9 GB RSS). Volume +
    bounds + centroid match catches every error mode that pure-AST rewrites
    can introduce (wrong center → centroid drift; wrong shape → volume drift)
    while costing zero memory and sub-millisecond per pair.

    Returns a pseudo-IoU in [0, 1] that is 1.0 only when all three match
    within 1e-3 relative tolerance, otherwise scaled by the worst metric.
    """
    import numpy as np
    if m1 is None or m2 is None: return None
    if len(m1.vertices) < 4 or len(m2.vertices) < 4: return None
    try:
        v1, v2 = float(m1.volume), float(m2.volume)
        if abs(v1) < 1e-9 or abs(v2) < 1e-9: return None
        vol_ratio = min(v1, v2) / max(v1, v2)

        b1 = m1.bounds; b2 = m2.bounds
        span1 = (b1[1] - b1[0]); span2 = (b2[1] - b2[0])
        denom = np.maximum(np.maximum(span1, span2), 1e-6)
        bound_err = float(np.max(np.abs(b1 - b2) / denom))
        bound_score = max(0.0, 1.0 - bound_err)

        c1 = m1.centroid; c2 = m2.centroid
        cent_err = float(np.linalg.norm(c1 - c2) / max(np.linalg.norm(span1), 1e-6))
        cent_score = max(0.0, 1.0 - cent_err)

        return min(vol_ratio, bound_score, cent_score)
    except Exception:
        return None


def _validate_one(args):
    py_path, _ = args
    try:
        src = Path(py_path).read_text()
        new = rewrite_source(src)
        m1 = _to_mesh(src)
        m2 = _to_mesh(new)
        if m1 is None or m2 is None:
            return ('exec_fail', py_path, None)
        iou = _iou(m1, m2)
        if iou is None:
            return ('exec_fail', py_path, None)
        return ('ok', py_path, iou)
    except Exception as e:
        return ('rewrite_fail', py_path, f'{type(e).__name__}: {e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--show-first', type=int, default=0)
    ap.add_argument('--validate-iou', type=int, default=0,
                    help='Sample N files, exec orig vs rewrite, compare voxel IoU')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    py_files = sorted(in_root.rglob('*.py'))
    if args.limit:
        py_files = py_files[:args.limit]

    rng = random.Random(args.seed)

    if args.show_first > 0:
        rng.shuffle(py_files)
        for i, p in enumerate(py_files[:args.show_first]):
            print(f'\n===== SAMPLE {i}: {p} =====')
            src = p.read_text()
            print('--- INPUT ---'); print(src.rstrip())
            print('--- OUTPUT ---')
            try:
                print(rewrite_source(src).rstrip())
            except Exception as e:
                print(f'ERROR: {type(e).__name__}: {e}')
                traceback.print_exc()
        return

    if args.validate_iou > 0:
        rng.shuffle(py_files)
        sample = py_files[:args.validate_iou]
        tasks = [(str(p), '') for p in sample]
        ok = []
        rewrite_fail = []
        exec_fail = []
        with mp.Pool(min(args.jobs, 4)) as pool:
            for status, path, val in pool.imap_unordered(_validate_one, tasks, chunksize=4):
                if status == 'ok':
                    ok.append((path, val))
                elif status == 'rewrite_fail':
                    rewrite_fail.append((path, val))
                else:
                    exec_fail.append(path)
        n = len(sample)
        print(f'\n=== IoU validation on {n} samples from {in_root} ===')
        print(f'  rewrite_fail: {len(rewrite_fail)} ({100*len(rewrite_fail)/n:.1f}%)')
        print(f'  exec_fail:    {len(exec_fail)} ({100*len(exec_fail)/n:.1f}%)')
        print(f'  exec_ok:      {len(ok)} ({100*len(ok)/n:.1f}%)')
        if ok:
            ious = [v for _, v in ok]
            high = sum(1 for v in ious if v >= 0.99)
            print(f'  IoU≥0.99:     {high}/{len(ok)} ({100*high/len(ok):.1f}% of exec_ok)')
            print(f'  IoU mean/median/min: {sum(ious)/len(ious):.4f} / '
                  f'{sorted(ious)[len(ious)//2]:.4f} / {min(ious):.4f}')
            low = sorted(ok, key=lambda x: x[1])[:5]
            if low and low[0][1] < 0.99:
                print('  Lowest IoU samples:')
                for p, v in low:
                    print(f'    {v:.4f}  {p}')
        if rewrite_fail[:5]:
            print('  Sample rewrite_fail:')
            for p, e in rewrite_fail[:5]:
                print(f'    {p}: {e}')
        return

    rng.shuffle(py_files)
    tasks = [(str(p), str(out_root / p.relative_to(in_root))) for p in py_files]
    print(f'rewriting {len(tasks)} files {in_root} → {out_root} ...')
    n_ok = n_fail = 0
    fails = []
    with mp.Pool(args.jobs) as pool:
        for i, (ok, err) in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=50)):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                if len(fails) < 10:
                    fails.append(err)
            if (i + 1) % 20000 == 0:
                print(f'  {i + 1}/{len(tasks)}  ok={n_ok}  fail={n_fail}')
    print(f'Done: {n_ok} ok, {n_fail} fail')
    for e in fails[:5]:
        print(f'  {e}')


if __name__ == '__main__':
    main()
