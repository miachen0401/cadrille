"""Regex-based feature extraction from CadQuery scripts.

No heavy dependencies — only `re` and `pathlib`.  This module is imported by
dataset_stats.py and failure_analysis.py; it can also be run standalone to
print a summary for a single script.

Usage:
    from scripts.analysis.parse_cq import parse_cq_script, load_cq_dir
    feats = parse_cq_script(open('some_script.py').read())

    # or load an entire directory
    records = load_cq_dir('./data/cad-recode-v1.5/train/batch_00')
"""

import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# Workplane plane type:  Workplane('XY') / Workplane('YZ',origin=...)
_PLANE_TYPE_RE    = re.compile(r"Workplane\s*\(\s*['\"]([A-Z]{2})['\"]")
# Any Workplane call (for counting)
_WORKPLANE_RE     = re.compile(r"Workplane\s*\(")
# Nested .workplane(offset=...) — secondary workplane
_WORKPLANE2_RE    = re.compile(r"\.workplane\s*\(")

# Sketch edge operations
_SEGMENT_RE       = re.compile(r"\.segment\s*\(")
_ARC_RE           = re.compile(r"\.arc\s*\(")
_SPLINE_RE        = re.compile(r"\.spline\s*\(")
_TANGENT_ARC_RE   = re.compile(r"\.tangentArcPoint\s*\(")
_THREE_PT_ARC_RE  = re.compile(r"\.threePointArc\s*\(")
_RADIUS_ARC_RE    = re.compile(r"\.radiusArc\s*\(")

# Sketch closed shapes
_CIRCLE_RE        = re.compile(r"\.circle\s*\(")
_RECT_RE          = re.compile(r"\.rect\s*\(")
_POLYGON_RE       = re.compile(r"\.polygon\s*\(")
_ELLIPSE_RE       = re.compile(r"\.ellipse\s*\(")
_SLOT_RE          = re.compile(r"\.slot\s*\(")

# Solid operations
_EXTRUDE_RE       = re.compile(r"\.extrude\s*\(")
_REVOLVE_RE       = re.compile(r"\.revolve\s*\(")
_LOFT_RE          = re.compile(r"\.loft\s*\(")
_SWEEP_RE         = re.compile(r"\.sweep\s*\(")

# Boolean operations
_UNION_RE         = re.compile(r"\.union\s*\(")
_CUT_RE           = re.compile(r"\.cut\s*\(|\.subtract\s*\(")
_INTERSECT_RE     = re.compile(r"\.intersect\s*\(")

# Primitives (no sketch needed)
_CYLINDER_RE      = re.compile(r"\.cylinder\s*\(")
_BOX_RE           = re.compile(r"\.box\s*\(")
_SPHERE_RE        = re.compile(r"\.sphere\s*\(")
_WEDGE_RE         = re.compile(r"\.wedge\s*\(")
_TORUS_RE         = re.compile(r"\.torus\s*\(")
_CONE_RE          = re.compile(r"\.cone\s*\(")

# Modifiers
_FILLET_RE        = re.compile(r"\.fillet\s*\(")
_CHAMFER_RE       = re.compile(r"\.chamfer\s*\(")
_SHELL_RE         = re.compile(r"\.shell\s*\(")
_MIRROR_RE        = re.compile(r"\.mirror\s*\(")
_ROTATE_RE        = re.compile(r"\.rotate\s*\(")

# Sketch flow
_PUSH_RE          = re.compile(r"\.push\s*\(")
_CLOSE_RE         = re.compile(r"\.close\s*\(")
_ASSEMBLE_RE      = re.compile(r"\.assemble\s*\(")
_FINALIZE_RE      = re.compile(r"\.finalize\s*\(")
_FACE_RE          = re.compile(r"\.face\s*\(")     # .face(...) inline sketch
_RESET_RE         = re.compile(r"\.reset\s*\(")

# Subtract mode in sketch: circle(r, mode='s') / rect(..., mode='s')
_SUBTRACT_MODE_RE = re.compile(r"mode\s*=\s*['\"]s['\"]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cq_script(code: str) -> Dict[str, Any]:
    """Extract structural features from a CadQuery script string.

    Returns a flat dict suitable for building a pandas DataFrame row.
    All counts are integers; boolean flags (has_*) are bool.
    """
    def _c(pattern):
        return len(pattern.findall(code))

    # Plane info
    planes       = _PLANE_TYPE_RE.findall(code)   # list of 'XY'/'YZ'/'ZX'
    n_wp         = _c(_WORKPLANE_RE)
    n_wp2        = _c(_WORKPLANE2_RE)

    # Sketch edge ops
    n_seg        = _c(_SEGMENT_RE)
    n_arc        = _c(_ARC_RE)
    n_spline     = _c(_SPLINE_RE)
    n_tan_arc    = _c(_TANGENT_ARC_RE)
    n_3pt_arc    = _c(_THREE_PT_ARC_RE)
    n_rad_arc    = _c(_RADIUS_ARC_RE)
    n_arcs_total = n_arc + n_tan_arc + n_3pt_arc + n_rad_arc

    # Sketch closed shapes
    n_circle     = _c(_CIRCLE_RE)
    n_rect       = _c(_RECT_RE)
    n_polygon    = _c(_POLYGON_RE)
    n_ellipse    = _c(_ELLIPSE_RE)
    n_slot       = _c(_SLOT_RE)

    # Solid ops
    n_extrude    = _c(_EXTRUDE_RE)
    n_revolve    = _c(_REVOLVE_RE)
    n_loft       = _c(_LOFT_RE)
    n_sweep      = _c(_SWEEP_RE)

    # Boolean ops
    n_union      = _c(_UNION_RE)
    n_cut        = _c(_CUT_RE)
    n_intersect  = _c(_INTERSECT_RE)

    # Primitives
    n_cylinder   = _c(_CYLINDER_RE)
    n_box        = _c(_BOX_RE)
    n_sphere     = _c(_SPHERE_RE)
    n_wedge      = _c(_WEDGE_RE)
    n_torus      = _c(_TORUS_RE)
    n_cone       = _c(_CONE_RE)

    # Modifiers
    n_fillet     = _c(_FILLET_RE)
    n_chamfer    = _c(_CHAMFER_RE)
    n_shell      = _c(_SHELL_RE)
    n_mirror     = _c(_MIRROR_RE)
    n_rotate     = _c(_ROTATE_RE)

    # Sketch flow
    n_push       = _c(_PUSH_RE)
    n_subtract_mode = _c(_SUBTRACT_MODE_RE)  # inline subtractions in sketch

    # Aggregates
    n_sketch_ops  = n_seg + n_arcs_total + n_spline + n_circle + n_rect + n_polygon + n_ellipse + n_slot
    n_bool_ops    = n_union + n_cut + n_intersect
    n_primitives  = n_cylinder + n_box + n_sphere + n_wedge + n_torus + n_cone
    n_modifiers   = n_fillet + n_chamfer + n_shell
    # Rough body count: each union/cut/intersect adds/removes a body
    n_bodies      = n_union + n_cut + n_intersect + 1

    return {
        # --- Plane info ---
        'planes':              planes,
        'n_plane_types':       len(set(planes)),
        'plane_xy':            planes.count('XY'),
        'plane_yz':            planes.count('YZ'),
        'plane_zx':            planes.count('ZX'),
        'n_workplanes':        n_wp,
        'n_secondary_wps':     n_wp2,

        # --- Sketch edge ops ---
        'n_segments':          n_seg,
        'n_arcs':              n_arcs_total,
        'n_arc_basic':         n_arc,
        'n_splines':           n_spline,

        # --- Sketch shapes ---
        'n_circles':           n_circle,
        'n_rects':             n_rects,
        'n_polygons':          n_polygon,
        'n_ellipses':          n_ellipse,

        # --- Solid ops ---
        'n_extrudes':          n_extrude,
        'n_revolves':          n_revolve,
        'n_lofts':             n_loft,
        'n_sweeps':            n_sweep,

        # --- Boolean ops ---
        'n_unions':            n_union,
        'n_cuts':              n_cut,
        'n_intersects':        n_intersect,

        # --- Primitives ---
        'n_cylinders':         n_cylinder,
        'n_boxes':             n_box,
        'n_spheres':           n_sphere,

        # --- Modifiers ---
        'n_fillets':           n_fillet,
        'n_chamfers':          n_chamfer,
        'n_shells':            n_shell,

        # --- Sketch flow ---
        'n_push':              n_push,
        'n_subtract_mode':     n_subtract_mode,

        # --- Aggregates ---
        'n_sketch_ops':        n_sketch_ops,
        'n_bool_ops':          n_bool_ops,
        'n_primitives':        n_primitives,
        'n_modifiers':         n_modifiers,
        'n_bodies':            n_bodies,
        'code_length':         len(code),

        # --- Boolean flags (presence) ---
        'has_arc':             n_arcs_total > 0,
        'has_spline':          n_spline > 0,
        'has_revolve':         n_revolve > 0,
        'has_loft':            n_loft > 0,
        'has_sweep':           n_sweep > 0,
        'has_cut':             n_cut > 0,
        'has_fillet':          n_fillet > 0,
        'has_chamfer':         n_chamfer > 0,
        'has_shell':           n_shell > 0,
        'has_cylinder':        n_cylinder > 0,
        'has_box':             n_box > 0,
        'has_sphere':          n_sphere > 0,
        'has_polygon':         n_polygon > 0,
        'has_push':            n_push > 0,
        'has_subtract_mode':   n_subtract_mode > 0,
        'has_multi_body':      n_bool_ops > 0,
    }


# Typo fix: n_rects key was mistyped above in the dict — fix here
def _fix_rects(d: dict, n_rect: int) -> dict:
    d['n_rects'] = n_rect
    return d


# Monkey-patch the above to pass n_rect correctly (parse_cq_script is a closure)
# instead do it cleanly by rewriting:

def parse_cq_script(code: str) -> Dict[str, Any]:  # noqa: F811 — intentional re-def
    """Extract structural features from a CadQuery script string."""
    def _c(pattern):
        return len(pattern.findall(code))

    planes           = _PLANE_TYPE_RE.findall(code)
    n_wp             = _c(_WORKPLANE_RE)
    n_wp2            = _c(_WORKPLANE2_RE)
    n_seg            = _c(_SEGMENT_RE)
    n_arc            = _c(_ARC_RE)
    n_spline         = _c(_SPLINE_RE)
    n_tan_arc        = _c(_TANGENT_ARC_RE)
    n_3pt_arc        = _c(_THREE_PT_ARC_RE)
    n_rad_arc        = _c(_RADIUS_ARC_RE)
    n_arcs_total     = n_arc + n_tan_arc + n_3pt_arc + n_rad_arc
    n_circle         = _c(_CIRCLE_RE)
    n_rect           = _c(_RECT_RE)
    n_polygon        = _c(_POLYGON_RE)
    n_ellipse        = _c(_ELLIPSE_RE)
    n_slot           = _c(_SLOT_RE)
    n_extrude        = _c(_EXTRUDE_RE)
    n_revolve        = _c(_REVOLVE_RE)
    n_loft           = _c(_LOFT_RE)
    n_sweep          = _c(_SWEEP_RE)
    n_union          = _c(_UNION_RE)
    n_cut            = _c(_CUT_RE)
    n_intersect      = _c(_INTERSECT_RE)
    n_cylinder       = _c(_CYLINDER_RE)
    n_box            = _c(_BOX_RE)
    n_sphere         = _c(_SPHERE_RE)
    n_fillet         = _c(_FILLET_RE)
    n_chamfer        = _c(_CHAMFER_RE)
    n_shell          = _c(_SHELL_RE)
    n_push           = _c(_PUSH_RE)
    n_subtract_mode  = _c(_SUBTRACT_MODE_RE)

    n_sketch_ops  = n_seg + n_arcs_total + n_spline + n_circle + n_rect + n_polygon + n_ellipse + n_slot
    n_bool_ops    = n_union + n_cut + n_intersect
    n_primitives  = n_cylinder + n_box + n_sphere
    n_bodies      = n_bool_ops + 1

    return {
        'planes':             planes,
        'n_plane_types':      len(set(planes)),
        'plane_xy':           planes.count('XY'),
        'plane_yz':           planes.count('YZ'),
        'plane_zx':           planes.count('ZX'),
        'n_workplanes':       n_wp,
        'n_secondary_wps':    n_wp2,
        'n_segments':         n_seg,
        'n_arcs':             n_arcs_total,
        'n_splines':          n_spline,
        'n_circles':          n_circle,
        'n_rects':            n_rect,
        'n_polygons':         n_polygon,
        'n_ellipses':         n_ellipse,
        'n_extrudes':         n_extrude,
        'n_revolves':         n_revolve,
        'n_lofts':            n_loft,
        'n_sweeps':           n_sweep,
        'n_unions':           n_union,
        'n_cuts':             n_cut,
        'n_intersects':       n_intersect,
        'n_cylinders':        n_cylinder,
        'n_boxes':            n_box,
        'n_spheres':          n_sphere,
        'n_fillets':          n_fillet,
        'n_chamfers':         n_chamfer,
        'n_shells':           n_shell,
        'n_push':             n_push,
        'n_subtract_mode':    n_subtract_mode,
        'n_sketch_ops':       n_sketch_ops,
        'n_bool_ops':         n_bool_ops,
        'n_primitives':       n_primitives,
        'n_bodies':           n_bodies,
        'code_length':        len(code),
        'has_arc':            n_arcs_total > 0,
        'has_spline':         n_spline > 0,
        'has_revolve':        n_revolve > 0,
        'has_loft':           n_loft > 0,
        'has_sweep':          n_sweep > 0,
        'has_cut':            n_cut > 0,
        'has_fillet':         n_fillet > 0,
        'has_chamfer':        n_chamfer > 0,
        'has_shell':          n_shell > 0,
        'has_cylinder':       n_cylinder > 0,
        'has_box':            n_box > 0,
        'has_sphere':         n_sphere > 0,
        'has_polygon':        n_polygon > 0,
        'has_push':           n_push > 0,
        'has_subtract_mode':  n_subtract_mode > 0,
        'has_multi_body':     n_bool_ops > 0,
    }


def load_cq_dir(data_dir: str, glob: str = '**/*.py',
                max_files: int = None) -> List[Dict[str, Any]]:
    """Load all CadQuery .py files from *data_dir*, parse each, return list of dicts.

    Skips files that don't contain 'cadquery' or 'cq.' (non-CQ scripts).
    """
    records = []
    paths = sorted(Path(data_dir).glob(glob))
    if max_files:
        paths = paths[:max_files]
    for path in paths:
        try:
            code = path.read_text(encoding='utf-8', errors='replace')
            if 'cadquery' not in code and 'cq.' not in code:
                continue
            feat = parse_cq_script(code)
            feat['path'] = str(path)
            feat['stem'] = path.stem
            feat['code'] = code
            records.append(feat)
        except Exception:
            pass
    return records


# ---------------------------------------------------------------------------
# Standalone: print summary for a single file
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python viz/parse_cq.py <script.py>')
        sys.exit(1)
    code = Path(sys.argv[1]).read_text()
    feats = parse_cq_script(code)
    for k, v in sorted(feats.items()):
        if k not in ('planes', 'code'):
            print(f'  {k:25s} {v}')
    print(f'  {"planes":25s} {feats["planes"]}')
