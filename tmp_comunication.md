最快的验证不是再大改方法，而是先做一个**最小三联实验**，把“问题到底在输入格式、初始化、还是数据泄漏/shape 分布”分开。

### 我建议直接做这 3 个 run

**Run A：RL init + GT only**
把你现在这套 LoRA recipe 几乎原样保留，只改输入：

* 不再用 hstack
* 只给 GT 图
* bad code 仍然给
* action 仍然给
* 输出 repaired code

这是最重要的主线，因为你们之前已经看到 pred render/hstack 大概率在干扰。

**Run B：RL init + 2-frame（GT + broken render 分开喂）**
不是 hstack，而是两张独立图。
这个 run 的作用不是追求最优，而是验证：

> pred render 本身到底有用，还是只是 hstack 这种格式有害。

如果 B 明显不如 A，结论就很清楚：**先彻底去掉 broken render**。

**Run C：SFT init + GT only**
这个只需要小规模跑，不用很久。
它回答的是：

> repair 能力主要来自 RL prior，还是其实普通 SFT init 也够。

---

## 这 3 个 run 的数据怎么配，才是“快速验证”

不要再沿用之前那个高重合的 val。
直接做一个**严格去重的小数据集**：

* train: 300–500 个 unique stems
* val: 50–100 个完全没见过的 stems
* test: 50–100 个完全没见过的 stems

而且 train/val/test 的 **target code 不能重复**。
哪怕数据量小，也要先保证这个结论是干净的。

更关键的是，train 的 GT program 不能再大部分都是最短最简单 shape。
要手工或规则化地补一些：

* rect
* circle
* ring
* L-shape
* multi-segment profile

不需要一次补很多，但至少别再让数据几乎全是最短模板。

**例子**：
如果 train 里 70% 都是 `rect/circle + single extrude`，那模型很容易学成：

* 看到 box corruption
* 输出一个合法 sketch 模板
  而不是学会：
* 根据 GT 图推断 profile 和尺寸

---

## 每个 run 只看 4 个指标

别再先盯 exact match。

1. **valid rate**
2. **mean ΔIoU**
3. **mean ΔCD**
4. **按 GT code length 分桶的 ΔIoU**

第四个最重要，因为它能直接告诉你：
模型是不是只会修简单 shape，一到复杂 profile 就崩。

可以简单分三桶：

* short: 最短 33%
* medium: 中间 33%
* long: 最长 33%

---

## 你现在这套 prompt，先怎么改

你现在是这种：

* 一张 hstack 图
* “Left half ... Right half ...”
* broken code
* assistant 输出 repair code

快速验证时，先改成下面这种更干净的格式：

### 版本 A：GT only

**[user]**

* 图像：只放 GT 4-view 图
* 文本：

  * “Target 3D shape is shown in the image.”
  * “Repair action: SWITCH_TO_SKETCH_EXTRUDE”
  * “Broken code: ...”

### 版本 B：2-frame

**[user]**

* 图像1：GT 4-view 图
* 图像2：broken code render 4-view 图
* 文本：

  * “First image: target shape.”
  * “Second image: current broken shape.”
  * “Repair action: SWITCH_TO_SKETCH_EXTRUDE”
  * “Broken code: ...”

先不要再写 “Left half / Right half” 这种依赖拼接布局的说明。

---

## 最快的决策规则

你其实只需要看这 3 个问题：

### 问题 1：A 能不能明显优于你之前的 hstack 方案？

如果 **RL + GT only** 已经比旧 hstack 好很多，说明主问题就是输入格式/干扰，不必纠结更多 fancy idea。

### 问题 2：B 比 A 好还是差？

* **B < A**：pred render 目前有害，后面先别碰
* **B ≈ A**：pred render 没价值，没必要加
* **B > A**：说明 pred render 有潜力，但以前是 hstack 格式毁了它

### 问题 3：C 和 A 差多少？

* **A > C 明显**：RL prior 值得保留，后面主线都用 RL init
* **A ≈ C**：说明 repair 主要靠新 SFT task，对 init 没那么敏感

---

## 我会怎么定义“路线过关”

这个阶段不用要求很夸张。
只要在**严格去重的真实 transfer test**上看到：

* valid rate ≥ 70%
* mean ΔIoU > 0
* mean ΔCD 改善
* 不是只在最短 code 桶有效

就说明路线是通的，可以扩到 600–800 GT programs。

如果结果是：

* valid rate 很高
* 但 ΔIoU 还是 ≤ 0
* ΔCD 也没改善

那就说明模型还是只学会了**合法 repair 模板**，没学会**几何修复**。

---

## 一句话的最快路线

**先做严格去重的小数据，直接跑 `RL+GT only`、`RL+2frame`、`SFT+GT only` 这三个对照；用 ΔIoU、ΔCD、按 code length 分桶结果判断，到底该保留 RL prior，还是彻底去掉 broken render。**

**例子**：
如果最后结果是

* A: ΔIoU = +0.03
* B: ΔIoU = -0.01
* C: ΔIoU = +0.01

那结论就非常清楚：
后面主线就是 **RL init + GT only**，不用再浪费时间在 hstack 和 broken render 上。
