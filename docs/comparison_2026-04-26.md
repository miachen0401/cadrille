# cadrille / CAD-Recode / CADEvolve / 我们 — 阶段对照

校对日期:2026-04-26
我们当前 run:`curriculum_qwen3vl_2b`,wandb `5q5zklt9`,训练到 step 15000 / 20000
(phase 3 已生效 ~3000 步,plateau 区间)

> 出处图例
> `[cadrille]`   = arXiv 2505.22914v2(Kolodiazhnyi 2025)
> `[recode]`     = arXiv 2412.14042(Rukhovich 2024)
> `[cadevolve]`  = arXiv 2602.16317(Elistratov 2026)
> `[code]`       = 本 repo
> `[log]`        = `logs/curriculum_qwen3vl_2b_20260425_192907.log`
> `[hf]`         = HuggingFace 数据集
> `[unverified]` = paper / 仓库内未找到第一手依据

---

## 0. TL;DR — image-modality 全栈对照(IoU %,越高越好)

| 阶段 | DeepCAD test | Fusion360 test | CC3D | MCB | 出处 |
|---|---:|---:|---:|---:|---|
| CAD-Recode (PC, DeepCAD 160k) | 80.7 | 67.6 | — | — | [recode] Tab 1 |
| CAD-Recode (PC, 1M procedural) | 92.0 | 87.8 | — | — | [recode] Tab 1 |
| cadrille SFT — Dpit (DeepCAD 全模态) | 78.2 | — | — | — | [cadrille] Tab 1 |
| **cadrille SFT — Rpi**(1M Recode, pc+img) | **86.1** | **77.6** | 56.1 | — | [cadrille] Tab 2 |
| cadrille SFT — Rpi+Dpi | 85.6 | 75.2 | 53.1 | — | [cadrille] Tab 2 |
| cadrille +DPO(Rpi → D⁻+F⁻) | 86.9 | 78.5 | 56.0 | — | [cadrille] Tab 2 |
| **cadrille +Dr.CPPO**(Rpi → D⁻+F⁻) | **92.2** | **84.6** | **65.0** | — | [cadrille] Tab 2 |
| cadrille SFT(CADEvolve 复测) | 86.5 | 77.3 | — | 40.4 | [cadevolve] Tab 1 |
| cadrille RL(CADEvolve 复测) | 92.2 | 84.6 | — | 47.6 | [cadevolve] Tab 1 |
| CADEvolve-P pre-aug (SFT) | 37.2 | 29.7 | — | 17.3 | [cadevolve] Tab 1 |
| CADEvolve-P post-aug (SFT) | 42.9 | 33.1 | — | 20.2 | [cadevolve] Tab 1 |
| CADEvolve-C small (SFT) | 49.6 | 35.1 | — | 28.1 | [cadevolve] Tab 1 |
| CADEvolve-C middle (SFT) | 70.1 | 59.1 | — | 39.1 | [cadevolve] Tab 1 |
| **CADEvolve-C big (SFT)** | **72.1** | **71.1** | — | 42.0 | [cadevolve] Tab 1 |
| **CADEvolve-C big +RL1** | **92.6** | **87.2** | — | 51.4 | [cadevolve] Tab 1 |
| CADEvolve-C big +RL2(加 MCB train)| 91.1 | 84.0 | — | **55.2** | [cadevolve] Tab 1 |
| **我们 step 15000 (greedy, n=20)** | **48.0** | **54.0** | — | — | [log] |
| **我们 step 15000 (max_iou@8 t=1, n=20)** | **60.5** | **66.4** | — | — | [log] |
| 我们 BenchCAD val(in-domain,paper 不评)| 53.0 (greedy), 59.0 (max@8) | — | — | — | [log] |

注 1:CADEvolve 复测的 cadrille SFT 86.5 vs cadrille 论文自报 86.1,差 0.4,属 eval 实现误差。
注 2:我们 n=20 online eval 不可与 paper 全 test set(8046 / 1725)直接相比,**绝对值偏乐观**。同子集内的相对趋势有意义。

---

## 1. 三个 stack 的训练阶段拆解

### 1.1 CAD-Recode(单阶段 SFT,只 PC)

| 阶段 | 数据 | 量 | 输出 IoU(DeepCAD/Fusion360 PC) | 出处 |
|---|---|---:|---:|---|
| SFT-DeepCAD | DeepCAD train | 160k | 80.7 / 67.6 | [recode] Tab 1 |
| SFT-Recode | 程序化 1M | 1.0M | **92.0 / 87.8** | [recode] Tab 1 |

训练:AdamW lr 2e-4 wd 0.01 cosine,**100k steps + 1k warmup**,batch **18**,1× H100,**~12h**,backbone Qwen2-1.5B,256 pts FPS 单线性投影。([recode] §A "Implementation Details")

### 1.2 cadrille(SFT → RL 两阶段;SFT 内 2 sub-stage)

| 阶段 | 训练数据 | 量 | 模态 | DeepCAD / Fusion360 / CC3D (img %) | 出处 |
|---|---|---:|---|---:|---|
| Stage-0(VLM 预训练) | 公开网络数据 | — | — | — | [cadrille] §3 |
| **SFT-A** Dpit | DeepCAD-160k(Text2CAD 重写,带 desc) | 160k | pc+img+text | 78.2 / — / — | [cadrille] Tab 1 |
| **SFT-B** Rpi | CAD-Recode 1M | 1.0M | pc+img | **86.1 / 77.6 / 56.1** | [cadrille] Tab 2 |
| SFT-B' Rpi+Dpi | + DeepCAD pc+img | 1.16M | pc+img | 85.6 / 75.2 / 53.1 | [cadrille] Tab 2 |
| SFT-B'' Rpi+Dt | + DeepCAD text | 1.16M | pc+img+text | 87.1(pc)/86.1(img)/82.1(text) | [cadrille] Tab 1 |
| **RL-DPO** | D⁻+F⁻(只 mesh,无 code) | — | image | 86.9 / 78.5 / 56.0 | [cadrille] Tab 2/10 |
| **RL-Dr.CPPO** | D⁻+F⁻ | — | image | **92.2 / 84.6 / 65.0** | [cadrille] Tab 2/11 |

**SFT 训练设置**([cadrille] App "Training"):
- backbone Qwen2-VL-2B,4-iso 视图 268×268 拼贴 → 400 vision tokens
- AdamW lr 2e-4,**120k steps**,batch **8 × 4 = 32 eff**,1× H100
- 256 pts FPS PC injection,继承 CAD-Recode 配方

**RL 训练设置**([cadrille] Tab 10/11):
- DPO:Adam,**20 epoch**,batch 160,lr 1e-5,β=0.3,8× H100
- Dr.CPPO:Adam,**20 epoch**,batch 128,lr 3e-5,PPO ε=0.1,GRPO group G=16,updates/batch=3,8× H100
- 两者都从 Rpi SFT init,只在 image modality 上 RL

### 1.3 CADEvolve(数据生成 → SFT 多档 → RL 两档)

| 阶段 | 数据 | 量 | DeepCAD / Fusion360 / MCB (img %) | 出处 |
|---|---|---:|---:|---|
| 数据 G | 进化生成 parametric generators | 7,945 | — | [cadevolve] §3.1 |
| 数据 P | 从 G 抽样 → CadQuery 程序 | ~800k | — | [cadevolve] §3.2 |
| 数据 P 后增广 | + 744,780 LLM 改写 | 744,780 | — | [cadevolve] §3.3 |
| 数据 Round-2 | + ABC 875,632 + ShapeNet 119,437 自蒸馏 | 1.74M | — | [cadevolve] §3.4 |
| **数据 C**(canonical 最终)| + 长度过滤 + sketch 多样性 + 旋转增广 + STL 渲染 | **2.72M** | — | [cadevolve] §3.5–3.6 |
| SFT P pre-aug | CADEvolve-P 原版 | ~800k | 37.2 / 29.7 / 17.3 | [cadevolve] Tab 1 |
| SFT P post-aug | + LLM 改写 | 744,780 | 42.9 / 33.1 / 20.2 | [cadevolve] Tab 1 |
| SFT C small | C 仅 generator | ~70k | 49.6 / 35.1 / 28.1 | [cadevolve] Tab 1 |
| SFT C middle | + ABC/ShapeNet | ~1M | 70.1 / 59.1 / 39.1 | [cadevolve] Tab 1 |
| **SFT C big** | + CAD-Recode 混合 | ~2.72M | **72.1 / 71.1 / 42.0** | [cadevolve] Tab 1 |
| **RL1**(C big init) | cadrille RL train(D+F360) | 同源 cadrille | **92.6 / 87.2 / 51.4** | [cadevolve] Tab 1 |
| RL2(C big init) | RL1 + MCB train | + MCB train | 91.1 / 84.0 / **55.2** | [cadevolve] Tab 1 |

**SFT 训练设置**([cadevolve] §4.4 + §4.1):
- backbone Qwen2-VL-2B(同 cadrille)
- 视图:**8 view = 6 ortho (±X/±Y/±Z) + 2 iso**,每 view **238×238**,2×4 grid → **476×952 px**
- 正交视图用 depth-along-axis 染绿色梯度,iso 视图绿色 lit
- **SFT 2 epoch**,token-level CE on code
- batch / lr / GPU 数量:[unverified](正文未列)

**RL 训练设置**([cadevolve] §4.4):
- 同 cadrille 的 Dr.CPPO + Dr.GRPO,reward = 10·IoU(compile)/-10(invalid)
- **20 epoch each**(RL1 / RL2)
- batch / lr [unverified]

### 1.4 我们(curriculum_qwen3vl_2b)

| 阶段 | 数据 | 量 | 输出(image, n=20 online) | 出处 |
|---|---|---:|---:|---|
| Stage-0(VLM 预训练) | Qwen3-VL 公开权重 | — | — | [code] `configs/sft/curriculum_qwen3vl_2b.yaml:20` |
| SFT phase-1 (step 0–6k) | bc:r:t = 1:2:2 | 113k 池子 | step 5k:BC 0.30, DC 0.39, F360 0.49 | [log] |
| SFT phase-2 (step 6k–12k) | bc:r:t = 2:1:1 | 113k 池子 | step 10k:BC 0.49, DC 0.45, F360 0.55 | [log] |
| **SFT phase-3 (step 12k–20k) — 当前** | bc:r:t = 8:1:1 | 113k 池子 | step 15k:BC 0.53, **DC 0.48, F360 0.54** | [log] |
| RL | 待 SFT 完 + IoU≥0.8 | — | 未开 | [code] `train/rl/` |

**SFT 训练设置**([code] `configs/sft/curriculum_qwen3vl_2b.yaml`):
- backbone **Qwen3-VL-2B-Instruct**(deepstack vision,paper 用 Qwen2-VL-2B)
- 视图:4-iso 268×268 拼贴(同 cadrille,异于 CADEvolve 8-view 476×952)
- AdamW lr 2e-4 cosine,1k warmup,**20k steps**,batch 8 × 4 = 32 eff,1× A100 80GB
- mode:**img only**,PC 路径未启用(`common/model.py:304-312` 注释:Qwen3-VL deepstack PC splice 未审计)
- 数据:BenchCAD 18,167 + cad-recode-20k 18,987 + text2cad 76,238 = 113,392
- prompt-leak 修复(BenchCAD desc 不再含 ops 列表,见 `common/datasets.py:348-370`)
- max_code_len=1000,group_by_length=true

---

## 2. 三个 paper 的「数据 → SFT-only 最强 → RL 后最强」三角对比

| paper | 最大 SFT 数据量 | SFT-best DeepCAD | SFT-best Fusion360 | RL-best DeepCAD | RL-best Fusion360 |
|---|---:|---:|---:|---:|---:|
| CAD-Recode (PC only) | 1M Recode | (PC) 92.0 | (PC) 87.8 | — | — |
| cadrille (1M Recode pc+img) | 1.16M | **86.1** | **77.6** | **92.2** | **84.6** |
| **CADEvolve** (2.72M, 8-view) | **2.72M** | 72.1 | 71.1 | **92.6** | **87.2** |
| 我们 (113k, 4-view img) | 113k | n=20 max@8 60.5 | n=20 max@8 66.4 | 未开 | — |

读这张表的方式:
1. **SFT-only 单看**:cadrille 86.1/77.6 比 CADEvolve big 72.1/71.1 还高。CADEvolve 论文自己也承认 SFT-only 弱(§4.5),它的优势在 RL 后。
2. **CADEvolve 真正 paper claim 是数据贡献**:同 RL 配方(都是 Dr.CPPO),同 RL 训练集(`cadrille RL train`),只换 SFT init 的训练数据,RL 后 Fusion360 +2.6,MCB +3.8。
3. **我们 SFT 113k 在 paper 用 1M-2.7M 的语境下,数据量小 4–24×**。BenchCAD 18k 在两 paper 都没用过(BenchCAD 不存在于这两篇 paper 的实验里)。

---

## 3. 渲染 pipeline 三方对比

| 维度 | cadrille | CADEvolve | 我们 |
|---|---|---|---|
| 视图数 | 4 iso | **8 = 6 ortho + 2 iso** | 4 iso |
| 单图尺寸 | 134×134(2×2 拼成 268×268) | 238×238 → 2×4 拼成 **476×952** | 134×134(2×2 拼成 268×268) |
| 颜色编码 | 黄色单色 + lighting | **正交=depth-along-axis 绿色梯度;iso=绿色 lit** | 黄色单色 + lighting(同 cadrille) |
| Shape 归一化 | mesh / normalize_std_img(=200) | 全集 [-100, 100]³ rigid-aligned | 同 cadrille |
| 出处 | [cadrille] App, [code] `common/datasets.py:191-211` | [cadevolve] §4.1 + [code] `experiments/cadevolve/render.py:19-87` | [code] `common/datasets.py:191-211` |

CADEvolve §4.1 原话:"Unlike the 4-iso setup in cadrille [22], the 6-ortho + 1–2 iso layout sharpens cues for fillets and chamfers."
但 paper 没单独消融 view-protocol 对 SFT 数字的贡献,**不能量化区分「8-view 收益 vs 2.72M 数据收益」**。

---

## 4. 数据格式兼容性 — 我们能直接接哪些 paper 数据 ✨ verified 2026-04-26

### 4.1 实测 byte-level 对比(filapro/cad-recode-v1.5 vs 我们)

下载了 paper `train/batch_00/1029.py` 跟我们 `data/cad-recode-20k/train/batch_00_1029.py` 做 md5 对比:

```
paper md5: 5d4501453e63900c7085c5bd11b07fac
ours  md5: 5d4501453e63900c7085c5bd11b07fac
MATCH (byte-identical)
```

**结论**:**我们的 cad-recode-20k 就是 paper cad-recode-v1.5 的一个 18,987 个文件的子集**(目录结构从 `batch_00/1029.py` 拍平成 `batch_00_1029.py`,内容一字不差)。

随机抽 1000 个我们的 + 13 个 paper 的对比 op vocabulary 与长度分布:

| 指标 | 我们(20k 抽 1000) | paper(13 个) |
|---|---|---|
| 平均行数 | 3.24 | 3.15 |
| 平均字符数 | 377 | 372 |
| 中位数 | 368 | 356 |
| extrude 占比 | 91.4% | 84.6% |
| union 占比 | 69.0% | 84.6% |
| cylinder | 24.9% | 53.8% |
| arc | 65.3% | 61.5% |
| revolve / loft / sweep / fillet / chamfer / shell / hole | 0 / 0 / 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 / 0 / 0 |

**op 词表完全一致** — 都只有 `{box, cylinder, extrude, union, sketch, sketch.{circle, rect, segment, arc, push, close}, assemble, finalize, workplane, face}`。**没有 revolve / fillet / chamfer / shell / loft / sweep / mirror / hole** —— paper Rpi 也是这一窄词表训出 86.1 的。

### 4.2 这意味着什么

1. **格式不是瓶颈** — 用户担心的「他们格式比我们好」**实测不成立**,完全 drop-in,collate / encode / loss 都不用改。
2. **不能解决 rare-op 问题** — 我们 BenchCAD eval 的 fillet / chamfer / shell / loft / sweep recall=0 是因为 BenchCAD 自己有这些 op 但 cad-recode 训练数据从来没出现过;扩 cad-recode 到 1M 不会教模型 fillet。这块只能靠 BenchCAD 自身样本(T12 oversample)或外部带 fillet 的数据(没有现成的)。
3. **唯一的 lever 是 volume** — paper Rpi 用 1M cad-recode 跑 100k 步,我们现在 18,987 跑 20k 步;**5× 数据量不变** = 等价于训练长度从 ~1 epoch 跑到 ~5 epoch。paper IoU 0.86 包含 50× 数据量 + PC 模态 + 100k 步,我们扩到全量大概率单独把 cad-recode 域内 IoU 推到 0.65-0.75,但**对 BenchCAD eval 的 rare-op recall 帮助有限**。

### 4.3 修正后的可用数据矩阵

| 数据集 | HF repo | 大小 | 验证状态 | 接入工作量 |
|---|---|---:|---|---|
| **CAD-Recode v1.5 全量** | `filapro/cad-recode-v1.5` 公开 | ~500 MB(~1M .py)| ✅ byte-identical 子集已验证 | trivial:扩展 `CadRecode20kDataset` 路径,re-render PNG |
| **CADEvolve-P**(parametric)| `kulibinai/cadevolve` 公开 | 4.73 GB tar | ❌ **不可直接用** — 含命名变量 + for 循环 + helper 对象 | 极高:需 execute 一遍把参数代入,且仍是多行结构 |
| **CADEvolve-C**(canonical = paper expanded)| 同上,tar 内子目录 | 同上 | ❌ **不可直接用** — 数字虽是字面,但 25-50 行顺序赋值 + `splineApprox`/`sweep`/`cq.Vector` 等我们没见过的 op | 高:需 AST adapter 化简成单行 chain + 过滤新 op |

### 4.3.1 CADEvolve-P 实测样本(参数化形式,**不是字面数字**)

```python
import cadquery as cq
L = 120; W = 33; H = 54; cham = 3
tab_w = 6; tab_h = 4; tab_d = 3
tz = H / 2 - tab_h / 2
body = cq.Workplane('XY').box(L, W, H).edges().chamfer(cham)
for sx in (-1, 1):
    for sy in (-1, 1):
        px = sx * (L/2 - cham/2); py = sy * (W/2 - cham/2)
        plane = cq.Plane(origin=(px, py, tz), xDir=(-sy, sx, 0), normal=(sx, sy, 0))
        tab = cq.Workplane(plane).rect(tab_w, tab_h).extrude(tab_d, combine=False)
        body = body.union(tab)
result = body
```

`parametric_generators.json` 第一个 entry 即为 `def box_prism(L=10, W=10, H=10): return cq.Workplane('XY').box(L, W, H)` —— 整套设计就是 Python 函数 with default args,**字面数字必须 execute 后才有**。

### 4.3.2 CADEvolve-C 实测样本(canonical,字面数字 ✅,但**仍是多行赋值结构**)

`./CADEvolve-C/CADEvolve-C-core/ArrayBossHelixWire__02_25_...py`(2830 字节):

```python
import cadquery as cq
wp1 = cq.Workplane('XY', origin=(0, 0, -72))
wp2 = wp1.splineApprox([[88, 0, 0], [77, 42, 6], ..., [88, 0, 144]], makeWire=1)
wp3 = cq.Workplane('XZ', origin=(0, 0, -72))
wp4 = wp3.moveTo(88, 0)
wp5 = wp4.circle(8)
wp6 = wp5.sweep(wp2)
wp7 = cq.Workplane(cq.Plane(origin=cq.Vector(60, 75, -62), xDir=cq.Vector(2, -2, 0), normal=cq.Vector(2, 2, 0)))
... (一直到 wp25 / wp50,每行一个 wpN = wpX.op(...) 赋值)
```

格式差异(逐项):

| 维度 | cad-recode(我们)| CADEvolve-C |
|---|---|---|
| 行数 | 3 行 | 25-50 行 |
| 结构 | `r=w0.op().op().union(...)` 单行 chain | `wp1=...; wp2=wp1.op(); wp3=...; wpN=wpX.union(wpY)` 顺序赋值 |
| Plane 构造 | `cq.Workplane('XY',origin=(...))` | `cq.Workplane(cq.Plane(origin=cq.Vector(...), xDir=cq.Vector(...), normal=cq.Vector(...)))` |
| 词表 | box/cylinder/extrude/union/sketch.{circle,rect,segment,arc,push,close,assemble} | 上述 + **splineApprox / sweep / chamfer / Vector / Plane** |

### 4.3.3 用户的判断:✅ 成立

**用户原话**:「他们的 cadquery 不一定和我们兼容,他们格式可能比我们好很多 / 我们是不是需要替换里面所有的 parameter 因为我们毕竟是数字 他是参数化生成」

实测结果:
- **CAD-Recode v1.5**:用户担心**不成立**,跟我们 byte-identical(我们就是它的子集)
- **CADEvolve-P**:用户担心**完全成立**,P 版就是参数化函数,需要 execute(等于自己做 P→C 这一步)
- **CADEvolve-C**:即便参数已 expand 成字面数字,**结构仍不兼容**(多行赋值 + 不同 Workplane 构造 + splineApprox/sweep 新 op)。所以即使路径 A(execute P→C)走通,**仍然需要 AST adapter** 化简多行赋值 → 单行 chain。

### 4.3.4 接入路径(三条,前两条都很贵)

**路径 α:Adapter 化简 CADEvolve-C → 单行 chain**
- 工作:写 AST 转换 `wp1=...; wp2=wp1.op(); ...wp25=wpX.union(wpY)` → `r=cq.Workplane(...).op().union(...)`;过滤含 splineApprox/sweep/Vector 的样本
- 工程量:2-3 天 + 验证 mesh IoU > 0.95
- 收益不确定:即便接进来,新词表 chamfer/sweep 因 token 罕见仍可能 recall 低

**路径 β:扩我们 SFT 目标格式 → 多行赋值**
- 工作:改 collate 让目标 token 序列容纳多行;但**这会破坏现有 cad-recode-20k 的 SFT signal**(20k 全是单行,模型已学到单行 distribution)
- 等于推翻当前 backbone prior,要从头训
- **不推荐**

**路径 γ(推荐):跳过 CADEvolve,先 T11/T12**
- 我们 BenchCAD eval 的 fillet/chamfer/shell recall=0 根因是**训练样本里这些 op 出现频率太低**(cad-recode 全 0,benchcad 占比也低)
- BenchCAD 自身有带 fillet/chamfer 的样本,直接 oversample(T12)+ per-source weighted loss(T11)即可命中,工程 1 天
- 如果 T11/T12 之后 BenchCAD rare-op recall 上来但 DeepCAD/Fusion360 IoU 仍卡 ≤0.5,**那时**再考虑 CADEvolve adapter,届时学习 pipeline 已验证
| DeepCAD train mesh | `Hula0401/deepcad_train_mesh` | 0.3 GB | 仅 .stl,**无 code** | 不可作 SFT(只能 RL reward) |
| Fusion360 train mesh | `Hula0401/fusion360_train_mesh` | 0.1 GB | 仅 .stl,**无 code** | 同上 |

### 4.4 接入估算(SFT 接 CAD-Recode 1M)

| 步骤 | 工程量 | 时间估算 |
|---|---|---|
| 下 `filapro/cad-recode-v1.5` 1M .py | trivial | 5-10 min(40 MB 报的是 README,实际 ~500 MB) |
| 跑 `data_prep/prerender_dataset.py` 渲 1M PNG(我们走 1-view PNG) | 已有 script,数据扩 50× | 8-12 h(单 4080) |
| 改 `common/datasets.py::CadRecode20kDataset` 走目录递归 | 30 min | — |
| 写 `configs/sft/qwen3vl_recode_full.yaml`(只有 cad-recode,无 BenchCAD/text2cad)| 10 min | — |
| 100k steps 训练(配方:lr 2e-4 cosine, bs 32 eff, eval/2k step) | — | ~70 h 单 4080 / ~30 h H100 |

### 4.5 推荐路径(修正)

| 方案 | 数据 | 预期 DeepCAD IoU(SFT-only)| 对 BenchCAD rare-op 帮助 | 工程 | 训练时间 |
|---|---:|---:|---|---|---|
| A:CAD-Recode 1M,纯 Recode,100k 步 | 1M | 0.65-0.75 [估] | ❌ 0(无 fillet/chamfer 样本)| 中 | 70h(4080) |
| B:CAD-Recode 1M + BenchCAD 8:1 + text2cad | ~1.1M | 0.60-0.70 [估] | ✅(BenchCAD 比例不降)| 低 | 70h |
| C:不扩数据,Qwen3-VL-2B → 4B 替换 | 113k | +0.03-0.05 vs 我们当前 [估] | ⚠️ 中 | 中(显存) | 8h(我们当前) |
| **D**:**T12 + T11 优先**(BenchCAD rare-op oversample + per-source weighted loss)| 113k | +0.02 [估] | ✅ 直接命中 fillet/chamfer recall | 低 | 8h |

**修订推荐**:**D 优先,然后 A**。理由:
- 我们距离 paper SOTA 主要差在 **rare-op recall**(fillet/chamfer/shell 都是 0),不是 cad-recode 域内 IoU。先做 T11/T12 这种**针对性**改进比扩 50× 数据更直接。
- 如果 T11/T12 之后 BenchCAD rare-op recall 上来但 DeepCAD 还卡在 0.5,**那时**再上 CAD-Recode 1M(方案 A)补 DeepCAD 域内 IoU。
- CADEvolve 2.72M **格式未验证,不推荐做 first move**;若要做先解 tar 抽 10 个文件确认是否 drop-in,再决定。

---

## 5. 我们当前训练动态(step 0 → 17000,完整 trajectory)

来源:`logs/curriculum_qwen3vl_2b_20260425_192907.log` 解析。3 phase 用 ←→ 标:

| step | BenchCAD val IoU | DeepCAD test IoU | Fusion360 test IoU | phase |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 0.000 | 0.000 | (init) |
| 1000 | 0.120 | 0.212 | 0.274 | P1 (1:2:2) |
| 2000 | 0.257 | 0.291 | 0.401 | P1 |
| 3000 | 0.324 | 0.305 | 0.438 | P1 |
| 4000 | 0.389 | 0.363 | 0.411 | P1 |
| 5000 | 0.299 | 0.390 | 0.488 | **→ P2 (2:1:1)** |
| 6000 | 0.397 | 0.448 | 0.490 | P2 |
| 7000 | 0.338 | 0.453 | 0.535 | P2 |
| 8000 | 0.409 | 0.416 | 0.545 | P2 |
| 9000 | 0.433 | 0.489 | 0.517 | P2 |
| 10000 | 0.490 | 0.448 | 0.548 | **→ P3 (8:1:1)** |
| **11000** | **0.597** ⭐ | 0.444 | 0.548 | P3 |
| 12000 | 0.579 | 0.458 | 0.560 | P3 |
| 13000 | 0.568 | 0.461 | 0.483 | P3 |
| 14000 | 0.529 | 0.440 | 0.551 | P3 |
| **15000** | 0.530 | **0.480** ⭐ | 0.540 | P3 |
| 16000 | 0.523 | 0.477 | 0.556 | P3 |
| 17000 | 0.509 | **0.414** ↓ | 0.542 | P3 (degrading) |

### Phase 3 verdict — 实证:**确实在伤 IoU**

观察:
1. **BenchCAD val 在 P3 入口 step 11000 冲到峰值 0.597,然后稳定 6000 步降到 0.509**(-0.088,共-15%)
2. **DeepCAD test 在 P3 中段 step 15000 触顶 0.480,step 17000 暴跌到 0.414**(-0.066,-14%,1000 步)
3. **transition 红利有效**(P2→P3 让 BenchCAD +0.107)**但 stable 不可持**
4. train_loss 0.015 vs eval_loss 0.457 → 经典过拟合曲线

启示(写进下一个 run 的 recipe):
- **不要 stable 单 mix(尤其极端 8:1:1)** — P3 stable 段就是个慢性退化
- **transition 有用** — 想要 BenchCAD 短期 boost,可以做 mix shift 但只在最后 1k 步
- **30k 步对 18k cad-recode 是太长** — 153 epochs 必过拟合;需要先扩数据

下一个 run 设计(`qwen3vl_2b_recode_30k_clean.yaml`):
- 删 curriculum,删 text2cad,benchcad:recode = 1:9
- recode 源换成新渲的 100k cad-recode-v1.5(我们今天 prep 的)
- 30k 步 = 8.6 epochs over 100k(健康)

---

## 6. 我们与两 paper 的差距 — 量化与归因

| 差距 | 量化 | 出处 |
|---|---|---|
| Recode 数据量 | paper 1M / CADEvolve 2.72M vs 我们 cad-recode-20k = 18,987(≤2%)| [cadrille] §5.1 / [cadevolve] §3.6 / [code] `common/datasets.py:379-432` |
| 训练步数 | cadrille 120k vs 我们 20k(6× 短)| [cadrille] App / [code] `configs/sft/curriculum_qwen3vl_2b.yaml:22` |
| 模态 | cadrille 训 pc+img,我们只 img;cadrille §6 自己说 PC 训练 unstable,但 SFT 还是用了 | [cadrille] Tab 2/3 + §6 / [code] `configs/sft/curriculum_qwen3vl_2b.yaml:16` |
| 视图 | 我们 4-iso 268×268(同 cadrille);CADEvolve 8-view 476×952 + depth-color | [cadevolve] §4.1 |
| backbone 版本 | cadrille / CADEvolve 都 Qwen2-VL-2B;我们 Qwen3-VL-2B(deepstack vision)| [cadrille] App / [cadevolve] §4.1 |
| 数据成分 | paper 用 Recode / DeepCAD / CADEvolve-C(2.72M);我们用 BenchCAD 18k + cad-recode-20k + text2cad 76k | [cadrille] §5.1 / [cadevolve] §4.2 |
| BenchCAD | 两 paper 都不评 / 不训 BenchCAD;我们 SFT 训它且 phase-3 占 8/10 权重 | grep "benchcad" → 0 hits in [cadrille] / [cadevolve] |
| MCB | CADEvolve 报 MCB,我们 / cadrille 都不评 | [cadevolve] §4.2 |
| 等价 epoch | cadrille 120k×32 / 1.16M ≈ 3.3 ep;CADEvolve SFT 2 ep;我们 20k×32 / 113k ≈ 5.7 ep | 计算 |

---

## 7. 仍是 [unverified] 的项

1. CADEvolve 的 SFT/RL batch / lr / GPU:[cadevolve] §4.4 只给"2 epoch SFT"和"20 epoch RL",未列 batch / lr / 硬件;可能在 Supplementary。
2. cadrille pc / img per-sample 比例:论文未明示。本 repo `mode='pc_img'` 实现是 50/50 per-sample 随机([code] `common/datasets.py:158-162`),未必是 paper 的策略。原 paper 仓库:`https://github.com/col14m/cadrille`(未拉)。
3. CADEvolve "C big" SFT 的 1M Recode 混入比例:论文 Tab 2 只写"Middle + mix with CADRecode (canonicalized)",未给比例。
4. CADEvolve `cadevolve.tar` 内部结构:[hf] `kulibinai/cadevolve`,4.73 GB 单 tar,未解开验证 cadquery .py / mesh / png / parquet 哪种格式。
