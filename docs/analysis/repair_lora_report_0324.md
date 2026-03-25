# Action-Conditioned CAD Repair LoRA — Experiment Report
**Date:** 2026-03-24
**Branch:** feat/work
**Author:** Cadrille research

---

## 1. 背景与目标

cadrille-rl 模型在 DeepCAD test set 上存在一类系统性失败：`wrong_primitive`，即模型预测了 `.box()` primitive 而非正确的 `sketch+extrude` 程序。本实验验证：能否通过 LoRA SFT 训练一个 repair 模块，在给定 GT 视图 + corrupt code + action token 的条件下，将 `.box()` 修复为正确的 sketch+extrude 程序。

---

## 2. 数据集构建

### 2.1 GT 程序来源

从 `data/analysis/deepcad_rl_img/` 中筛选 cadrille-rl 在 DeepCAD test set 上的**高质量预测**：
- 筛选条件：IoU ≥ 0.95，且代码使用 `sketch+extrude`（无 `.box()`）
- 来源文件：`data/analysis/deepcad_rl_img/{stem}_pred.py`
- 初始筛选：200 个 unique stems

### 2.2 Corruption 生成

脚本：`tools/gen_repair_data.py --n 200`

每个 stem 生成两种 corruption：
- **type1**：bbox 精确匹配的 `box()`（执行 GT code → 提取 BoundingBox → 生成 `box(xsize, ysize, zsize)`）
- **type2**：bbox ±15% 随机扰动的 `box()`

每种 corruption 同时渲染 corrupt render PNG（`data/repair_sft/corrupt_renders/{stem}_{type}.png`）。

### 2.3 数据过滤（v2，本次实验）

发现以下类型 case 不符合任务定义，予以过滤：

| 过滤条件 | 数量 | 原因 |
|---------|------|------|
| `.cylinder()` primitive | 10 | GT 使用 primitive，非 sketch+extrude，与 action 矛盾 |
| multi-extrude (≥2) | 19 | 单个 box() → 多特征重建，难度不一致，非"repair"而是重建 |
| 合计 | 29 | |

过滤后：**171 unique stems**，400 → **342 pairs**（每 stem 2 个 corruption type）。

### 2.4 Train/Val Split

**v1（错误，已废弃）**：随机按 pair 分割 → 32/36 val stems 在 train 中出现（不同 corruption type）→ val_loss 衡量的是 memorization，无效。

**v2（本次实验）**：stem-level 分割，随机 seed=42：
- Train：136 stems × 2 types = **272 pairs**
- Val：35 stems × 2 types = **70 pairs**
- Stem overlap：**0**

数据集 profile 分布（train）：
- segment: 120/171 (70%)
- circle: 81/171 (47%)
- rect: 47/171 (27%)
- arc: 15/171 (9%)
- Workplane: XY 98, ZX 87, YZ 15

---

## 3. 模型与训练设置

### 3.1 Base Model

`checkpoints/cadrille-rl`：cadrille-sft 经过 CPPO RL 训练后的模型（Qwen2-VL-2B backbone）。

**注意**：RL 训练时输入格式为 `"Generate cadquery code"` + 单张 268×268 GT render，与 repair SFT 输入格式有显著分布差异。

### 3.2 LoRA 配置

```
r=16, alpha=32, dropout=0.0
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
trainable params: 18,464,768 / 2,227,530,240 (0.83%)
```

### 3.3 训练超参数

```
lr = 1e-4 (cosine decay with warmup 10%)
epochs = 10
batch_size = 1, grad_accum = 8 (effective batch = 8)
optimizer = AdamW, weight_decay = 0.01
max_new_tokens (eval) = 1024
```

### 3.4 输入格式（v2，hstack）

```
[system] You are a helpful assistant.
[user]
  <|video_pad|> × 190   ← 536×268 hstack（左：GT 4-view | 右：corrupt render 4-view）
  "Left half: target 3D shape (4 views). Right half: current broken prediction (4 views).
   Repair action: SWITCH_TO_SKETCH_EXTRUDE — the box() fallback must be replaced with
   a proper sketch+extrude pattern matching the target geometry.
   Rewrite the code using sketch+extrude.

   Broken code:
   import cadquery as cq
   r = cq.Workplane('XY').workplane(offset=...).moveTo(...).box(...)"
[assistant]
  "import cadquery as cq
   w0=cq.Workplane(...)
   r=w0.sketch()...extrude(...)"
```

Token 统计（单 example）：total 442，visual 190，supervised 92（21%）。

### 3.5 W&B Run

`repair-lora-r16-lr1e-04-ep10-hstack`
https://wandb.ai/hula-the-cat/cadrille-rl/runs/ssh6ikt2

---

## 4. 训练结果

### 4.1 Run A: hstack（v2，本次主实验）

W&B: `repair-lora-r16-lr1e-04-ep10-hstack` / run `ssh6ikt2`

| Epoch | train_loss | val_loss |
|-------|-----------|---------|
| 1 | 0.5601 | 0.3294 |
| 2 | 0.2381 | 0.2743 |
| **3** | **0.1761** | **0.2695 ← best** |
| 4 | 0.1296 | 0.3041 |
| 5 | 0.0871 | 0.4007 |
| 6 | 0.0640 | 0.5025 |
| 7 | 0.0418 | 0.5928 |
| 8 | 0.0283 | 0.6689 |
| 9 | 0.0187 | 0.6986 |
| 10 | 0.0136 | 0.7446 |

### 4.2 Run B: 2frame

W&B: `repair-lora-r16-lr1e-04-ep10-2frame` / run `hqt0whyk`

| Epoch | train_loss | val_loss |
|-------|-----------|---------|
| 1 | 0.3566 | 0.2644 |
| **2** | **0.1852** | **0.2264 ← best** |
| 3 | 0.1426 | 0.2331 |
| 4 | 0.1068 | 0.2661 |
| 5 | 0.0776 | 0.3242 |
| 6 | 0.0530 | 0.4022 |
| 7 | 0.0383 | 0.5005 |
| 8 | 0.0237 | 0.5455 |
| 9 | 0.0161 | 0.6101 |
| 10 | 0.0114 | 0.6571 |

**两个 run 共同结论**：epoch 2-3 后严重过拟合，train_loss 趋近于 0，val_loss 单调上升。136 stems 对 LoRA r=16 来说数据量严重不足。2frame 的 best val_loss（0.2264）略优于 hstack（0.2695），收敛更快（epoch 2 vs epoch 3）。

---

## 5. 迁移评估（Transfer Eval）

### 5.1 评估设置

脚本：`tools/eval_repair_lora.py`
评估集：`data/analysis/deepcad_rl_img/` 中**真实** wrong_primitive cases（与训练数据完全独立）：
- 筛选：IoU ∈ [0.30, 0.88]，代码包含 `.box()`（`is_wrong_primitive()` 检测）
- n=50，IoU 范围 [0.465, 0.735]，baseline mean IoU = 0.6281

后处理：`fix_trailing_paren()`（剥离末尾多余 `)` ，修复 tokenizer 的 `))` token artifact）

评估时输入格式与训练保持一致（`--input-mode` 参数）。

### 5.2 结果：hstack (Run A)

| Checkpoint | valid rate | mean ΔIoU | median ΔIoU | ΔIoU>0.05 |
|-----------|-----------|----------|------------|----------|
| epoch_01 | 78% | -0.521 | -0.552 | 0% |
| epoch_03 (best) | 92% | -0.510 | -0.572 | 0% |
| epoch_05 | 92% | -0.499 | -0.581 | 0% |

### 5.3 结果：2frame (Run B)

| Checkpoint | valid rate | mean ΔIoU | median ΔIoU | ΔIoU>0.05 |
|-----------|-----------|----------|------------|----------|
| epoch_01 | 84% | -0.583 | — | 0% |
| epoch_02 (best) | 88% | -0.587 | — | 0% |
| epoch_03 | 86% | -0.556 | — | 0% |

### 5.4 Run A vs Run B 综合对比

| | best val_loss | best valid rate | best mean ΔIoU | ΔIoU>0.05 |
|---|---|---|---|---|
| hstack | 0.2695 (ep3) | 92% | -0.499 | **0%** |
| 2frame | 0.2264 (ep2) | 88% | -0.556 | **0%** |

**结论：两种输入格式结果无显著差异，均失败。** 2frame 在 val_loss 上略好，但 transfer eval 上 ΔIoU 反而更差（-0.556 vs -0.499）。

### 5.5 典型输出示例

```python
# 输入 corrupt code（典型）
r = cq.Workplane('XY').workplane(offset=-83.5).moveTo(0,0).box(176.8, 138.8, 167.0)

# 模型输出（两种格式均类似）
import cadquery as cq
w0=cq.Workplane('XY',origin=(0,0,1))
r=w0.sketch().push([(0,0)]).rect(200,200).finalize().extrude(2)
# → 合法代码，但几何完全不对（200×200×2 placeholder）
```

---

## 6. Bug 记录与修复

### 6.1 Trailing `))` Token Artifact ✅ 已修复

**现象**：约 8-22% 的输出结尾多出一个 `)`，导致 SyntaxError，valid rate=0%。
**原因**：Qwen2 tokenizer 中 `))` 是单独 token（id=593）。SFT teacher forcing 不执行代码，无法通过 reward 惩罚；RL 训练因 reward=0（执行失败）天然消除此问题。
**修复**：`fix_trailing_paren()` 后处理加入 `eval_repair_lora.py`（2026-03-24）。

### 6.2 eval_repair_lora.py 缺少 --input-mode 参数 ✅ 已修复

**现象**：2frame 模型用 hstack 格式评估，输入分布不匹配。
**修复**：加入 `--input-mode hstack/2frame/gt-only` 参数，`build_item()` 按模式构建输入（2026-03-24）。

### 6.3 Val Set Stem 泄漏（v1）✅ 已修复

**现象**：v1 val set 中 32/36 stems 在 train 中出现（不同 corruption type）→ val_loss 衡量 memorization，best checkpoint 选择无效。
**修复**：v2 改为 stem-level 分割，seed=42，136/35 stems，overlap=0（2026-03-24）。

### 6.4 脏数据（cylinder primitive + multi-extrude）✅ 已修复

**现象**：10 个 `.cylinder()` case（GT 非 sketch+extrude，与 action 矛盾）+ 19 个 multi-extrude case（单 box→多特征，难度不一致）。
**修复**：过滤后保留 171 stems（2026-03-24）。

---

## 7. 失败分析

### 7.1 现象

模型学会了输出"合法的 sketch+extrude 代码"，但几何内容与 GT 图像无关——输出固定是小尺寸 placeholder（如 `rect(200,200).extrude(2)`）。

### 7.2 根本原因

1. **视觉条件未被有效利用**：hstack 格式（536×268）包含 190 个 visual token，但模型在 RL 预训练时只见过 268×268 单图（100 visual token），输入分布偏移大。
2. **任务定义与监督信号不对齐**：cross-entropy loss 仅优化 token 对齐，不惩罚几何错误；92 个 supervised token 里几何关键信息（origin 坐标、尺寸数值）所占比例低。
3. **数据量不足**：136 stems，3 epoch 内即过拟合，不足以学到 visual → geometry 的泛化。
4. **corrupt render 可能是干扰**（未定论，ablation 结果因数据泄漏问题待重测）：hstack 中的 corrupt render 可能反而把模型注意力带向错误几何。

---

## 8. 下一步实验

| Run | 输入格式 | 命令 | 目的 |
|-----|---------|------|------|
| **Run A** | GT only | `--input-mode gt-only` | 去掉 corrupt render 干扰，验证最简情形 |
| **Run B** | 2-frame | `--input-mode 2frame` | GT + corrupt 分帧，避免 hstack 位置歧义 |

两个 run 评估后对比 ΔIoU，决定是否继续扩数据（目标 300-500 stems）。

---

## 9. 复现命令

```bash
# 数据生成（原始 200 stems）
python3 tools/gen_repair_data.py --n 200 --out data/repair_sft

# 数据过滤 + stem-level split 重建（v2）
# 过滤规则：去掉 .cylinder() primitive 和 n_extrude>=2
# split: seed=42, 136 train / 35 val stems
# → 详见 tools/gen_repair_data.py 或直接用已生成的 train.jsonl / val.jsonl

# Run A: hstack
PYTHONUNBUFFERED=1 python3 tools/train_repair_lora.py \
    --checkpoint checkpoints/cadrille-rl \
    --train-data data/repair_sft/train.jsonl \
    --val-data   data/repair_sft/val.jsonl \
    --out        checkpoints/repair-lora-v2 \
    --epochs 10 --lr 1e-4 --lora-rank 16 --input-mode hstack

# Run B: 2frame
PYTHONUNBUFFERED=1 python3 tools/train_repair_lora.py \
    --checkpoint checkpoints/cadrille-rl \
    --train-data data/repair_sft/train.jsonl \
    --val-data   data/repair_sft/val.jsonl \
    --out        checkpoints/repair-lora-2frame \
    --epochs 10 --lr 1e-4 --lora-rank 16 --input-mode 2frame

# 评估（input-mode 必须与训练一致）
python3 tools/eval_repair_lora.py \
    --checkpoint checkpoints/repair-lora-v2/epoch_03 \
    --base-model checkpoints/cadrille-rl \
    --n 50 --input-mode hstack --out data/repair_eval_v2/epoch_03

python3 tools/eval_repair_lora.py \
    --checkpoint checkpoints/repair-lora-2frame/epoch_02 \
    --base-model checkpoints/cadrille-rl \
    --n 50 --input-mode 2frame --out data/repair_eval_2frame/epoch_02
```

---

## 10. 文件索引

| 路径 | 说明 |
|------|------|
| `data/repair_sft/train.jsonl` | 272 pairs，136 stems（v2，过滤后） |
| `data/repair_sft/val.jsonl` | 70 pairs，35 stems（v2，0 overlap） |
| `data/repair_sft/corrupt_renders/` | 600 PNG：{stem}\_gt / \_type1 / \_type2 |
| `checkpoints/repair-lora-v2/` | Run A hstack，epoch_01~10 + best（=epoch_03） |
| `checkpoints/repair-lora-2frame/` | Run B 2frame，epoch_01~10 + best（=epoch_02） |
| `data/repair_eval_v2/epoch_{01,03,05}/results.json` | Run A eval 结果 |
| `data/repair_eval_2frame/epoch_{01,02,03}/results.json` | Run B eval 结果 |
| `tools/train_repair_lora.py` | 训练脚本（--input-mode hstack/2frame/gt-only） |
| `tools/eval_repair_lora.py` | 评估脚本（fix_trailing_paren + --input-mode） |
| `tools/gen_repair_data.py` | 数据生成脚本 |
| `docs/analysis/repair_lora_report_0324.md` | 本报告 |
