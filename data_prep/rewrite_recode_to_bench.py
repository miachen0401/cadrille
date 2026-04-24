"""Reformat cad-recode-v1.5 .py files into BenchCAD-style shell.

Transforms compact single-line recode style:

    import cadquery as cq
    w0=cq.Workplane('ZX',origin=(0,20,0))
    r=w0.sketch().segment(...).extrude(-40).union(w0.workplane(...).box(...))

Into multi-line `result = (...)` style (matches BenchCAD/cad_bench targets):

    import cadquery as cq

    result = (
        cq.Workplane('ZX', origin=(0, 20, 0))
        .sketch()
        .segment(...)
        .extrude(-40)
        .union(
            cq.Workplane('ZX', origin=(0, 20, 0))
            .workplane(...)
            .box(...)
        )
    )

    # Export
    show_object(result)

Pure AST rewrite — no CadQuery execution, so semantic preservation relies on:
  - Inlining middle Workplane vars (w0, w1, ...) via NodeTransformer substitution
  - Custom pretty-printer that breaks method chains onto separate lines

Usage:
  # Smoke test with 3 samples printed to stdout
  uv run python tools/rewrite_recode_to_bench.py \\
      --input data/cad-recode-v1.5/train --output /tmp/out --show-first 3

  # Full batch rewrite (parallel)
  uv run python tools/rewrite_recode_to_bench.py \\
      --input data/cad-recode-v1.5/train \\
      --output data/cad-recode-v1.5-bench/train --jobs 8
"""
import argparse
import ast
import copy
import multiprocessing as mp
import traceback
from pathlib import Path


INDENT = '    '


def is_chain(node):
    """Call on an Attribute target (i.e. x.method(...))."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)


def chain_length(node):
    """Number of method calls chained on `node`. 0 for non-chains."""
    n = 0
    cur = node
    while is_chain(cur):
        n += 1
        cur = cur.func.value
    return n


def split_chain(node):
    """Walk outermost → innermost Call.Attribute chain.

    Returns (base_node, [(method_name, args, keywords), ...] in inner→outer order).
    """
    methods = []
    cur = node
    while is_chain(cur):
        methods.append((cur.func.attr, cur.args, cur.keywords))
        cur = cur.func.value
    return cur, list(reversed(methods))


def _unparse(node):
    return ast.unparse(node)


def render_args(args, keywords, indent):
    """Render an argument list.

    Inline if all args are short (chain_length < 2);
    multi-line if any arg is itself a long method chain (e.g. union/cut payloads).
    """
    break_pos = [chain_length(a) >= 2 for a in args]
    break_kw = [chain_length(kw.value) >= 2 for kw in keywords]
    if not any(break_pos) and not any(break_kw):
        parts = [_unparse(a) for a in args] + \
                [f'{kw.arg}={_unparse(kw.value)}' for kw in keywords]
        return ', '.join(parts)

    inner_indent = INDENT * (indent + 1)
    outer_indent = INDENT * indent
    pieces = []
    for a, brk in zip(args, break_pos):
        if brk:
            rendered = render_chain(a, indent + 1)
            # render_chain returns: first line (base, no prefix) + '\n' + subsequent
            # .method lines already prefixed with INDENT*(indent+1). Add inner_indent
            # to the first line only.
            first, _, rest = rendered.partition('\n')
            piece = inner_indent + first + ('\n' + rest if rest else '')
        else:
            piece = inner_indent + _unparse(a)
        pieces.append(piece)
    for kw, brk in zip(keywords, break_kw):
        if brk:
            rendered = render_chain(kw.value, indent + 1)
            first, _, rest = rendered.partition('\n')
            # inner_indent for kw prefix, rest keeps its own indent
            piece = inner_indent + f'{kw.arg}=' + first + ('\n' + rest if rest else '')
        else:
            piece = inner_indent + f'{kw.arg}={_unparse(kw.value)}'
        pieces.append(piece)

    return '\n' + ',\n'.join(pieces) + '\n' + outer_indent


def render_chain(node, indent):
    """Render a possibly-chain expression.

    Convention: the first line (the "base") has no leading indent — the caller
    places it. Subsequent `.method(...)` lines are prefixed with INDENT*indent.
    """
    base, methods = split_chain(node)
    if not methods:
        return _unparse(node)

    # Merge first method into base_txt when base is a plain Name (e.g. `cq.Workplane(...)`)
    if isinstance(base, ast.Name):
        first_name, first_args, first_kw = methods[0]
        args_txt = render_args(first_args, first_kw, indent)
        base_txt = f'{base.id}.{first_name}({args_txt})'
        methods = methods[1:]
    elif is_chain(base):
        base_txt = render_chain(base, indent)
    else:
        base_txt = _unparse(base)

    out_lines = [base_txt]
    for name, args, kw in methods:
        args_txt = render_args(args, kw, indent)
        out_lines.append(f'{INDENT * indent}.{name}({args_txt})')
    return '\n'.join(out_lines)


def inline_vars(expr, var_map):
    """Replace Name(x) with a deep copy of var_map[x] throughout `expr`."""
    class Inliner(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in var_map:
                return copy.deepcopy(var_map[n.id])
            return n
    return Inliner().visit(copy.deepcopy(expr))


def rewrite_source(src):
    """Transform recode-style source → bench-shell source.

    Raises ValueError on unsupported inputs (e.g. no assignment target).
    """
    tree = ast.parse(src)

    var_map = {}
    final_target = None
    final_value = None
    had_import = False

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            had_import = True
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            tgt = stmt.targets[0].id
            var_map[tgt] = stmt.value
            final_target = tgt
            final_value = stmt.value
            continue
        # Other top-level statements (print, exec, etc.) — not expected in recode
        raise ValueError(f'Unsupported top-level stmt: {type(stmt).__name__}')

    if final_target is None:
        raise ValueError('No assignment target found')
    if not had_import:
        # Fine — we always emit import; just note it
        pass

    # Resolve chains between middle vars (w1 = w0.workplane() style) to fixed point
    middle_vars = {k: v for k, v in var_map.items() if k != final_target}
    max_iters = len(middle_vars) + 1
    for _ in range(max_iters):
        changed = False
        for k in list(middle_vars.keys()):
            others = {kk: vv for kk, vv in middle_vars.items() if kk != k}
            new_val = inline_vars(middle_vars[k], others)
            if ast.dump(new_val) != ast.dump(middle_vars[k]):
                middle_vars[k] = new_val
                changed = True
        if not changed:
            break

    inlined = inline_vars(final_value, middle_vars)

    body_txt = render_chain(inlined, indent=1)
    # First line needs its own indent too (inside the `result = (...)` block)
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
# Batch driver
# ---------------------------------------------------------------------------

def _process_one(args):
    in_path, out_path = args
    try:
        src = Path(in_path).read_text()
        new_src = rewrite_source(src)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(new_src)
        return True, None
    except Exception as e:
        return False, f'{in_path}: {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=str, required=True,
                    help='Input root dir (e.g. data/cad-recode-v1.5/train)')
    ap.add_argument('--output', type=str, required=True,
                    help='Output root dir (e.g. data/cad-recode-v1.5-bench/train)')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most N files (for smoke test)')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--show-first', type=int, default=0,
                    help='Print first N rewrites to stdout for visual inspection (single-process)')
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    py_files = sorted(in_root.rglob('*.py'))
    if args.limit:
        py_files = py_files[:args.limit]
    print(f'Found {len(py_files)} .py files under {in_root}')

    tasks = [(str(p), str(out_root / p.relative_to(in_root))) for p in py_files]

    if args.show_first > 0:
        for i, (ip, op) in enumerate(tasks[:args.show_first]):
            print(f'\n===== SAMPLE {i}: {ip} =====')
            src = Path(ip).read_text()
            print('--- INPUT ---')
            print(src.rstrip())
            print('--- OUTPUT ---')
            try:
                print(rewrite_source(src).rstrip())
            except Exception as e:
                print(f'ERROR: {type(e).__name__}: {e}')
                traceback.print_exc()
        return

    n_ok = 0
    n_fail = 0
    fails = []
    with mp.Pool(args.jobs) as pool:
        for i, (ok, err) in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=50)):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                if len(fails) < 20:
                    fails.append(err)
            if (i + 1) % 20000 == 0:
                print(f'  {i + 1}/{len(tasks)}  ok={n_ok}  fail={n_fail}')
    print(f'\nDone: {n_ok} ok, {n_fail} fail')
    if fails:
        print('Sample failures:')
        for e in fails[:10]:
            print(f'  {e}')


if __name__ == '__main__':
    main()
