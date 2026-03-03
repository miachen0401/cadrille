"""Fillet gap analysis and visualization.

Creates:
  1. Synthetic CadQuery example showing box with vs without fillet
  2. Arc-as-fillet workaround example (2D rounded corner via arc in sketch)
  3. Side-by-side render of a real test pair (GT vs our prediction)
  4. Summary explanation of why fillets are absent and what the impact is

Run:
    python viz/fillet_analysis.py
"""

import os, sys, json, textwrap, subprocess, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT_DIR = 'viz/plots/fillet_analysis'
os.makedirs(OUT_DIR, exist_ok=True)


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


# ---------------------------------------------------------------------------
# Execute CadQuery code in subprocess, return trimesh or None
# ---------------------------------------------------------------------------

_WORKER = textwrap.dedent('''\
import sys, json, traceback
import numpy as np
code = sys.stdin.read()
try:
    import cadquery as cq
    import trimesh
    g = {}
    exec(code, g)
    r = g["r"]
    verts, faces = r.val().tessellate(0.001, 0.1)
    v = [(v.x, v.y, v.z) for v in verts]
    f = [list(tri) for tri in faces]
    print(json.dumps({"verts": v, "faces": f}))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
''')

def _exec_cq(code: str):
    fd, path = tempfile.mkstemp(suffix='.py')
    with os.fdopen(fd, 'w') as f:
        f.write(_WORKER)
    try:
        proc = subprocess.run(
            [sys.executable, path],
            input=code, capture_output=True, text=True, timeout=30
        )
        if proc.stdout.strip():
            d = json.loads(proc.stdout.strip())
            if 'error' in d:
                print(f'  CQ error: {d["error"]}')
                return None
            import trimesh as _tm
            return _tm.Trimesh(d['verts'], d['faces'])
    except Exception as e:
        print(f'  exec failed: {e}')
        return None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Matplotlib 3D mesh renderer (headless)
# ---------------------------------------------------------------------------

def _render_mesh(ax, mesh, color='steelblue', alpha=0.85, title=''):
    import trimesh as _tm
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Simplify if huge
    if len(faces) > 3000:
        ratio = 3000 / len(faces)
        try:
            mesh = mesh.simplify_quadric_decimation(int(len(faces) * ratio))
            verts = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
        except Exception:
            pass

    poly = Poly3DCollection(verts[faces], alpha=alpha,
                            facecolor=color, edgecolor='none', linewidth=0)
    # Shade by face normal z-component for depth cue
    normals = mesh.face_normals
    shade   = 0.5 + 0.5 * np.clip(normals[:, 2], -1, 1)
    poly.set_facecolor([plt.cm.Blues(s) for s in shade])
    ax.add_collection3d(poly)

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mid = (lo + hi) / 2
    ext = max(hi - lo) / 2 * 1.1
    ax.set_xlim(mid[0] - ext, mid[0] + ext)
    ax.set_ylim(mid[1] - ext, mid[1] + ext)
    ax.set_zlim(mid[2] - ext, mid[2] + ext)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=25, azim=135)
    if title:
        ax.set_title(title, fontsize=10, pad=4)


# ---------------------------------------------------------------------------
# Plot 1: Box without fillet vs Box with fillet (synthetic)
# ---------------------------------------------------------------------------

CODE_NO_FILLET = textwrap.dedent('''\
import cadquery as cq
r = (cq.Workplane("XY")
       .box(1.0, 0.6, 0.4))
''')

CODE_WITH_FILLET = textwrap.dedent('''\
import cadquery as cq
r = (cq.Workplane("XY")
       .box(1.0, 0.6, 0.4)
       .edges("|Z")
       .fillet(0.08))
''')

CODE_ARC_APPROX = textwrap.dedent('''\
import cadquery as cq
# Rounded rectangle profile using arcs in 2D sketch
# approximates a "2D fillet" on extruded cross-section
r = (
    cq.Workplane("XY")
      .sketch()
      .rect(1.0, 0.6)
      .vertices()
      .fillet(0.08)
      .finalize()
      .extrude(0.4)
)
''')

CODE_ARC_MANUAL = textwrap.dedent('''\
import cadquery as cq, math
# Manual arc-based rounded corner — what the model CAN generate
R = 0.08
W, H, D = 1.0, 0.6, 0.4
r = (
    cq.Workplane("XY")
      .moveTo(R, 0)
      .lineTo(W - R, 0)
      .radiusArc((W, R), -R)
      .lineTo(W, H - R)
      .radiusArc((W - R, H), -R)
      .lineTo(R, H)
      .radiusArc((0, H - R), -R)
      .lineTo(0, R)
      .radiusArc((R, 0), -R)
      .close()
      .extrude(D)
)
''')


def plot_fillet_comparison():
    print('Generating CQ meshes for fillet comparison ...')
    mesh_no  = _exec_cq(CODE_NO_FILLET)
    mesh_yes = _exec_cq(CODE_WITH_FILLET)
    mesh_arc = _exec_cq(CODE_ARC_MANUAL)

    fig = plt.figure(figsize=(15, 5), facecolor='white')
    fig.suptitle('Fillet gap: what the model can vs cannot generate', fontsize=13, y=1.01)

    # --- Left: no fillet ---
    ax1 = fig.add_subplot(131, projection='3d')
    if mesh_no:
        _render_mesh(ax1, mesh_no, title=f'Model output (no fillet)\nn_faces={len(mesh_no.faces)}')
    else:
        ax1.set_title('No-fillet mesh\n(CQ exec failed)')
    ax1.set_title('Model output\n(sharp edges, what model generates)', fontsize=9)

    # --- Middle: with fillet ---
    ax2 = fig.add_subplot(132, projection='3d')
    if mesh_yes:
        _render_mesh(ax2, mesh_yes, title=f'GT with .fillet(0.08)\nn_faces={len(mesh_yes.faces)}')
    else:
        ax2.set_title('Fillet mesh\n(CQ exec failed)')
    ax2.set_title('GT ideal output\n(.fillet(0.08) rounded edges — never in training)', fontsize=9)

    # --- Right: arc workaround ---
    ax3 = fig.add_subplot(133, projection='3d')
    if mesh_arc:
        _render_mesh(ax3, mesh_arc, title=f'Arc workaround\nn_faces={len(mesh_arc.faces)}')
    else:
        ax3.set_title('Arc workaround\n(CQ exec failed)')
    ax3.set_title('Arc-based workaround\n(radiusArc rounds 2D corners only)', fontsize=9)

    # Annotations
    for ax, label, color in [
        (ax1, '✗ sharp corners', '#F44336'),
        (ax2, '✓ smooth 3D edges', '#4CAF50'),
        (ax3, '~ 2D corners rounded only', '#FF9800'),
    ]:
        ax.text2D(0.5, -0.05, label, transform=ax.transAxes,
                  ha='center', color=color, fontsize=9, fontweight='bold')

    fig.tight_layout()
    _savefig(fig, '01_fillet_comparison_3d.png')


# ---------------------------------------------------------------------------
# Plot 2: 2D sketch — arc vs fillet explanation
# ---------------------------------------------------------------------------

def plot_arc_vs_fillet_2d():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    fig.suptitle('How arcs relate to fillets in CadQuery', fontsize=13)

    # --- Left: sharp corner (no fillet, no arc) ---
    ax = axes[0]
    ax.set_aspect('equal')
    W, H = 1.0, 0.6
    rect = plt.Polygon([(0,0),(W,0),(W,H),(0,H)], fill=True,
                       facecolor='#BBDEFB', edgecolor='steelblue', linewidth=2)
    ax.add_patch(rect)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.7)
    ax.set_title('Sharp corners\n(model output — what it generates)', fontsize=10)
    ax.annotate('Sharp 90°\ncorner', xy=(W, H), xytext=(0.75, 0.75),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=8)
    ax.text(0.5, -0.05, 'CQ: .rect(W, H).extrude(D)',
            ha='center', fontsize=8, transform=ax.transAxes, style='italic')
    ax.axis('off')

    # --- Middle: 2D arc in sketch (rounds cross-section corners) ---
    ax = axes[1]
    ax.set_aspect('equal')
    R = 0.08
    # Draw rounded rectangle via arc approximation
    import matplotlib.patches as mpatches
    fancy = mpatches.FancyBboxPatch((0, 0), W, H,
                                     boxstyle=f'round,pad=0,rounding_size={R}',
                                     facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(fancy)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.7)
    ax.annotate('Arc rounds\n2D corner', xy=(W, 0), xytext=(0.75, -0.05),
                arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=8)
    ax.set_title('2D arc in sketch\n(rounds extruded cross-section only)', fontsize=10)
    ax.text(0.5, -0.12,
            'CQ: .radiusArc(..., -R)  ← model CAN learn this',
            ha='center', fontsize=8, transform=ax.transAxes, style='italic',
            color='#388E3C')
    ax.axis('off')

    # --- Right: 3D fillet on solid (rounds all selected edges) ---
    ax = axes[2]
    ax.set_aspect('equal')
    # Draw box outline with rounded-looking corners in 2D projection
    fancy2 = mpatches.FancyBboxPatch((0, 0), W, H,
                                      boxstyle=f'round,pad=0,rounding_size={R}',
                                      facecolor='#FFE0B2', edgecolor='#E65100', linewidth=2)
    ax.add_patch(fancy2)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.7)
    # Show "3D" depth lines
    for x, y in [(0,0),(W,0),(W,H),(0,H)]:
        ax.plot([x, x+0.08], [y, y+0.06], color='#E65100', linewidth=1, alpha=0.5)
    ax.annotate('3D edge\nrounded', xy=(W/2, H), xytext=(0.2, 0.9),
                arrowprops=dict(arrowstyle='->', color='#E65100'), color='#E65100', fontsize=8,
                xycoords='data', textcoords=('data', 'axes fraction'))
    ax.set_title('3D fillet on solid\n(.fillet() — 0% in training, model never generates)', fontsize=10)
    ax.text(0.5, -0.12,
            'CQ: .edges("|Z").fillet(R)  ← ABSENT from training data',
            ha='center', fontsize=8, transform=ax.transAxes, style='italic',
            color='#E65100')
    ax.axis('off')

    fig.tight_layout()
    _savefig(fig, '02_arc_vs_fillet_2d.png')


# ---------------------------------------------------------------------------
# Plot 3: Real GT vs prediction pair
# ---------------------------------------------------------------------------

def plot_real_pair(stem='00000093'):
    import trimesh as _tm
    gt_path   = f'data/deepcad_test_mini/{stem}.stl'
    pred_path = f'work_dirs/eval_cadrille_sft/{stem}+0.py'
    pred_stl  = f'work_dirs/tmp_pair/{stem}+0.stl'

    if not os.path.exists(gt_path):
        print(f'  GT not found: {gt_path}')
        return

    # Execute prediction .py to get mesh
    os.makedirs('work_dirs/tmp_pair', exist_ok=True)
    if os.path.exists(pred_path) and not os.path.exists(pred_stl):
        code = open(pred_path).read()
        mesh = _exec_cq(code)
        if mesh:
            mesh.export(pred_stl)

    gt_mesh   = _tm.load_mesh(gt_path)
    pred_mesh = _tm.load_mesh(pred_stl) if os.path.exists(pred_stl) else None

    fig = plt.figure(figsize=(14, 5), facecolor='white')

    # Left: GT
    ax1 = fig.add_subplot(121, projection='3d')
    _render_mesh(ax1, gt_mesh)
    ax1.set_title(f'GT mesh — {stem}\n(from DeepCAD test set)\n'
                  f'faces={len(gt_mesh.faces)}, watertight={gt_mesh.is_watertight}',
                  fontsize=9)

    # Right: prediction
    ax2 = fig.add_subplot(122, projection='3d')
    if pred_mesh:
        _render_mesh(ax2, pred_mesh, color='darkorange')
        ax2.set_title(f'Our prediction (Cadrille-SFT)\n'
                      f'faces={len(pred_mesh.faces)}, watertight={pred_mesh.is_watertight}',
                      fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Prediction mesh\nnot available', transform=ax2.transAxes,
                 ha='center', va='center')
        ax2.set_title('Our prediction (failed)', fontsize=9)

    fig.suptitle(f'Real test example: {stem}  (GT vs Cadrille-SFT prediction)', fontsize=12)
    fig.tight_layout()
    _savefig(fig, f'03_real_pair_{stem}.png')


# ---------------------------------------------------------------------------
# Plot 4: CadQuery code snippet comparison
# ---------------------------------------------------------------------------

def plot_code_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')

    code_can = '''\
# What the model CAN generate
import cadquery as cq
r = (
  cq.Workplane("XY")
    .box(1.0, 0.6, 0.4)
    .union(
      cq.Workplane("XY")
        .cylinder(0.3, 0.2)
    )
)
# → sharp edges only
# → no rounded transitions
'''

    code_cannot = '''\
# What the model CANNOT generate (0% in training)
import cadquery as cq
r = (
  cq.Workplane("XY")
    .box(1.0, 0.6, 0.4)
    .union(
      cq.Workplane("XY")
        .cylinder(0.3, 0.2)
    )
    .edges("|Z")      # select vertical edges
    .fillet(0.05)     # ← THIS LINE IS NEVER IN TRAINING
    .edges(">>Z")     # select top face edges
    .chamfer(0.03)    # ← NOR THIS
)
# → smooth, realistic CAD shapes
# → matches real-world Fusion360/SolidWorks output
'''

    for ax, code, color, title in [
        (axes[0], code_can,    '#E3F2FD', 'What model generates\n(sharp edges)'),
        (axes[1], code_cannot, '#FFF3E0', 'What real CAD needs\n(fillets — never learned)'),
    ]:
        ax.set_facecolor(color)
        ax.text(0.02, 0.98, code, transform=ax.transAxes,
                fontfamily='monospace', fontsize=8.5,
                va='top', ha='left', wrap=True,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    axes[0].text(0.5, 0.02, '✓ Training data: 91.5% use extrude, 70.6% use union',
                 transform=axes[0].transAxes, ha='center', color='#1565C0', fontsize=9)
    axes[1].text(0.5, 0.02, '✗ Training data: 0% use .fillet(), 0% use .chamfer()',
                 transform=axes[1].transAxes, ha='center', color='#E65100', fontsize=9)

    fig.suptitle('Fillet gap: CadQuery operations absent from training distribution\n'
                 '(model trained on CAD-Recode v1.5, which never uses .fillet() or .chamfer())',
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, '04_code_comparison.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Generating fillet analysis plots → {OUT_DIR}/')

    print('\n[1/4] 3D mesh comparison (no fillet / with fillet / arc workaround) ...')
    plot_fillet_comparison()

    print('\n[2/4] 2D sketch: arc vs fillet explanation ...')
    plot_arc_vs_fillet_2d()

    print('\n[3/4] Real test pair GT vs prediction ...')
    # Use highest-face-count shape (most likely to show differences)
    for stem in ['00000093', '00010900', '00005807']:
        plot_real_pair(stem)

    print('\n[4/4] Code snippet comparison ...')
    plot_code_comparison()

    print('\nDone.')
