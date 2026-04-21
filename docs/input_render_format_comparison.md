# 模型输入渲染格式对比

## 总览

| 属性 | Cadrille (SFT/RL) | CadEvolve |
|------|-------------------|-----------|
| **视角数量** | 4 | 8 |
| **视角类型** | 对角线方向 | 6正交 + 2等轴 |
| **排列方式** | 2×2 网格（单张图，作为 video 传入） | 4×2 网格（单张图，作为 image 传入） |
| **图像分辨率** | 268×268 px | 476×952 px |
| **渲染器** | open3d `Visualizer(visible=False)` | PyVista `off_screen=True` |
| **着色方式** | 均匀黄色 + Phong 光照 | 坐标编码绿色通道（正交视角） / 纯绿色（等轴视角）|
| **背景颜色** | 白色（tile内部），黑色（tile边框2px） | 黑色 |
| **系统 prompt** | 有（生成CadQuery代码指令） | 无 |
| **输入 token 类型** | `video` (fps=1.0) | `image` |

---

## Cadrille — 4视角详情

```
fronts = [[1,1,1], [-1,-1,-1], [-1,1,-1], [1,-1,1]]
camera_distance = -0.9
img_size = 128 (per tile) → 268×268 composite (含2px黑色tile边框)
```

网格布局（2×2）：

| 位置 | front 方向 | 语义描述 |
|------|-----------|---------|
| 左上 (0,0) | [1, 1, 1] | 右前上对角 |
| 右上 (0,1) | [-1,-1,-1] | 左后下对角 |
| 左下 (1,0) | [-1, 1,-1] | 左前下对角 |
| 右下 (1,1) | [1,-1, 1] | 右后上对角 |

**注意**：4个视角是互补对角方向（覆盖8个卦限中的4个，均匀分布）。

---

## CadEvolve — 8视角详情

```
tile_size = 14 * 17 = 238 px
collage = 476 × 952 px (2列 × 4行)
mesh normalized to [0,1]^3
```

网格布局（4行×2列）：

| 行 | 左列 (col=0) | 右列 (col=1) | 类型 |
|----|-------------|-------------|------|
| 0  | **-Z**（俯视，view_xy neg, flip=True） | **+Z**（仰视，view_xy, no flip） | 正交，zoom=1.7 |
| 1  | **+Y**（view_xz neg, flip=True） | **-Y**（view_xz, no flip） | 正交，zoom=1.7 |
| 2  | **+X**（view_yz, flip=True） | **-X**（view_yz neg, no flip） | 正交，zoom=1.7 |
| 3  | **Iso**（view_isometric, flip=True） | **-Iso**（negative isometric, no flip） | 透视，zoom=1.1 |

着色：正交视角用**坐标编码绿色**（沿该视角轴坐标 → G通道强度），等轴用纯绿色+光照。

---

## bench 评测输入与训练输入的对齐分析

### Cadrille

| 属性 | 训练时 | bench 评测时（`eval_bench.py`）| 是否对齐 |
|------|--------|-------------------------------|---------|
| 图像来源 | `render_img(stl_path)` via open3d | HF `composite_png`（同为open3d，但有橙/红 **edge highlight**） | ⚠️ 部分对齐 |
| 视角顺序 | [1,1,1], [-1,-1,-1], [-1,1,-1], [1,-1,1] | 同上（2×2网格） | ✅ |
| 分辨率 | 268×268 | 268×268 | ✅ |
| 背景 | 白色tile + 黑色border | 白色tile + 黑色border | ✅ |
| Edge highlight | ❌ 无 | ✅ 有橙色/红色 edge | ⚠️ **domain shift** |
| 输入格式 | video | video | ✅ |

**结论**：Cadrille bench 评测存在轻微 domain shift（edge highlight），但视角顺序、分辨率、布局完全一致。

### CadEvolve

| 属性 | 训练时 | bench 评测时（`experiments/cadevolve/eval.py`）| 是否对齐 |
|------|--------|-------------------------------|---------|
| 图像来源 | GT STL → PyVista 8-view | GT code → STL → PyVista 8-view（`render_stl()`） | ✅ |
| 视角顺序 | -Z, +Z, +Y, -Y, +X, -X, Iso, -Iso | 同上 | ✅ |
| 分辨率 | 476×952 | 476×952 | ✅ |
| 着色 | 坐标绿色 | 坐标绿色 | ✅ |
| 系统 prompt | 无 | 无 | ✅ |
| 输入格式 | image（单张composite） | image（单张composite） | ✅ |

**结论**：CadEvolve bench 评测与训练格式完全对齐。

---

## 可视化对比（同一 piston 样本）

左：bench composite_png（Cadrille 评测时的实际输入，含橙色 edge highlight）
右：Cadrille training render（训练时的图像格式，纯 open3d shading）

格式差异很小，主要区别在于 edge highlight。

---

## 结论 & 建议

1. **CadEvolve 评测格式是正确的** — 与训练完全一致
2. **Cadrille 评测有轻微 domain shift**（edge highlight）—— 若要严格公平评测，应用 `render_img()` 重新渲染 GT，而不是直接用 HF 的 `composite_png`
3. **Cadrille bench IoU 低（~0.1）的主要原因不是渲染格式**，而是模型本身的性能（训练数据是 DeepCAD/Fusion360，benchmark 是合成件）
