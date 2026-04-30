# Paper training-section metrics plan — 2026-04-30

Source: in-session brainstorm with user (2026-04-30). Captures the
"what to add to the training story for the paper" plan + framework so
the writeup is *credible* and *interpretable*, not just chasing more
absolute IoU.

Status: framework runs end-to-end and reaches ~80% of strongest baseline.
Now we shift from "push the number" to "make the result defensible to
reviewers." Minimum 4 additions below.

---

## 1. Same-setup retrain with held-out families

**Required.** Goal isn't to bump the score — it's to answer the most
likely reviewer question:

> Has SFT/RL learned CAD operations, or just memorized part templates?

### Recipe

Keep everything fixed:

```text
same model
same data size if possible
same LR
same epochs / steps
same prompt
same eval script
```

Only change the split:

```text
Train: remove 2–4 families
Test:  evaluate on the removed families
```

Suggested removals (must be representative):

| Removed family       | Probes                                          |
| -------------------- | ----------------------------------------------- |
| twist drill          | helix / sweep / flute                           |
| involute gear        | involute / circular array / tooth profile       |
| handwheel            | spoke / revolve / circular pattern              |
| one simpler family   | sanity control                                  |

Expected pattern:

```text
Seen-family test:   SFT/RL improves a lot
Unseen-family test: improvement is smaller
```

Conclusion to write:

> CAD-specific training improves template-level and operation-level
> ability, but unseen industrial-part generalization remains hard.

---

## 2. Rare / advanced ops recall

**Required.** Most distinguishing metric in the training section.

### Op partition

```python
BASIC_OPS = {
    "line", "polyline", "circle", "rectangle",
    "extrude", "cut",
}

COMMON_ADVANCED_OPS = {
    "revolve", "fillet", "chamfer", "mirror", "array",
}

RARE_OPS = {
    "loft", "sweep", "helix", "thread",
    "involute", "gear_tooth_profile",
    "twisted_extrude", "shell",
}
```

### Three metrics

**Op Recall** — how many GT-required ops the prediction contains.

```text
Op Recall = |Pred Ops ∩ GT Ops| / |GT Ops|
```

**Advanced Op Recall** — restricted to revolve/fillet/chamfer/array/mirror.

**Rare Op Recall** — restricted to helix/sweep/loft/involute/thread, the
ops that actually distinguish industrial CAD.

### Expected pattern

| Model |       Op Recall | Advanced Op Recall | Rare Op Recall |
| ----- | --------------: | -----------------: | -------------: |
| Base  |             low |          very low  |     near zero  |
| SFT   |          higher |             higher |       improves |
| RL    | similar/higher  |             higher |        highest |

Even if rare-op recall is absolutely small, *relative* improvement is
the story.

---

## 3. Valid Rare Op Recall

Standard rare-op recall is attackable:

> The model just emitted the string `.sweep()`, but the code doesn't run.

So gate on executability:

```text
Valid Rare Op Recall = rare op recall computed only on executable outputs

  if code fails to execute → rare op recall = 0
  if code executes         → count rare op overlap as usual
```

This pairs naturally with the executable-CAD-code framing.

### Final table

| Model | Exec Pass | Op Recall | Rare Op Recall | Valid Rare Op Recall |
| ----- | --------: | --------: | -------------: | -------------------: |
| Base  |           |           |                |                      |
| SFT   |           |           |                |                      |
| RL    |           |           |                |                      |

---

## 4. Operation diversity / collapse

We need a metric that explicitly answers "did the model collapse to a
handful of ops?" — defends against the "just memorized" critique.

### Unique Valid Ops (primary)

```text
# of distinct CAD ops the model used across all executable outputs
```

| Model | Unique Valid Ops |
| ----- | ---------------: |
| Base  |                5 |
| SFT   |               12 |
| RL    |               15 |
| GT    |               20 |

### Operation Entropy (secondary, appendix)

```text
H = -Σ p(op) log p(op)
```

> Is the model only doing extrude/cut, or is its op distribution rich?

If short on time, ship Unique Valid Ops in the main table and entropy in
appendix.

---

## Minimum final metric list

| Metric                       | Required? |
| ---------------------------- | --------- |
| Exec Pass                    | yes       |
| IoU / CD                     | yes       |
| Op Recall                    | yes       |
| Rare Op Recall               | yes       |
| Valid Rare Op Recall         | yes       |
| Unique Valid Ops             | yes       |
| Seen vs Unseen-family gap    | yes       |

That's it. Don't add more.

---

## Result tables

### Table A — same setup, original split

| Model  | Exec ↑ | IoU ↑ | CD ↓ | Op Recall ↑ | Rare Op Recall ↑ | Valid Rare Op Recall ↑ | Unique Valid Ops ↑ |
| ------ | -----: | ----: | ---: | ----------: | ---------------: | ---------------------: | -----------------: |
| Base   |        |       |      |             |                  |                        |                    |
| SFT    |        |       |      |             |                  |                        |                    |
| SFT+RL |        |       |      |             |                  |                        |                    |

### Table B — unseen-family split

| Model | Split          | Exec ↑ | IoU ↑ | Op Recall ↑ | Rare Op Recall ↑ | Valid Rare Op Recall ↑ |
| ----- | -------------- | -----: | ----: | ----------: | ---------------: | ---------------------: |
| Base  | seen-family    |        |       |             |                  |                        |
| SFT   | seen-family    |        |       |             |                  |                        |
| RL    | seen-family    |        |       |             |                  |                        |
| Base  | unseen-family  |        |       |             |                  |                        |
| SFT   | unseen-family  |        |       |             |                  |                        |
| RL    | unseen-family  |        |       |             |                  |                        |

Table B is the most important. The conclusion:

> SFT/RL improves CAD-specific operation use on seen families,
> especially rare operation recall. However, gains are reduced on
> held-out part families, indicating that industrial CAD generation
> still requires better compositional generalization rather than
> template memorization.

---

## Optional metrics (only 2 worth adding)

### A. Feature Count Accuracy

Many parts are not IoU-determined but feature-count-determined:

| Part      | Feature           |
| --------- | ----------------- |
| gear      | tooth count       |
| handwheel | spoke count       |
| drill     | flute count       |
| flange    | hole count        |
| bracket   | slot / hole count |

```text
Feature Count Accuracy = (predicted key feature count == GT count)
Feature Count Error    = |pred_count - gt_count|
```

Useful because shape can be close but `gear teeth = wrong`,
`handwheel spokes = wrong`, `flange holes = wrong` — IoU misses these.

### B. Parameter Relative Error (only if metadata is reliable)

Diameter, hole radius, tooth number, spoke count.
**Only do this if a stable parser exists.** Otherwise skip.

---

## Priority queue

### Priority 0 — lock the eval first

```text
Save Base / SFT / RL generated outputs:
  generated code
  exec status
  IoU
  GT ops
  pred ops
```

(We already do this via `predictions/step-NNNNNN.jsonl` and
`predictions/step-NNNNNN.max@K.jsonl`. Done.)

### Priority 1 — unseen-family retrain

Remove 2–4 families, retrain SFT (and RL if time permits). RL on the new
SFT can be a short run.

### Priority 2 — ops metrics

```text
Op Recall
Advanced Op Recall
Rare Op Recall
Valid Rare Op Recall
Unique Valid Ops
```

### Priority 3 — feature count

```text
gear tooth count
handwheel spoke count
flange hole count
drill flute count
```

Coverage doesn't need to be exhaustive.

---

## Story (English + Chinese)

> Although the trained model reaches only about 80% of the strongest
> baseline on overall geometry score, it substantially improves
> CAD-specific operation metrics. In particular, SFT/RL increases rare
> operation recall and the number of valid advanced operations used in
> executable programs. When evaluated on held-out part families, the
> gains decrease, suggesting that current models learn useful CAD
> construction patterns but still struggle with compositional
> generalization beyond seen industrial templates.

> 分数不是最强,但它确实学到了 CAD 里的关键 operation;不过一换没
> 见过的零件族,提升变小,说明工业 CAD 不是简单刷数据就能解决。这个
> 结果反过来证明我们的 benchmark 有价值。

---

## Modification suggestions (Claude review, 2026-04-30)

**1. Family-removal list — verify against the actual training corpus first.**
We need to grep `family` field across:

  - `data/benchcad/train.pkl` (11k items)
  - `data/cad-iso-106/train.pkl` (122k items, industrial parts — most
    likely place to find drill / gear / handwheel)
  - `data/benchcad-simple/train.pkl` (77k items)
  - `data/text2cad-bench/train.pkl` (53k items)
  - `data/cad-recode-bench/train.pkl` (472k items)

Some of the suggested removals (`twist_drill`, `involute_gear`,
`handwheel`) may not actually appear as named families in our data. If
they don't exist as families, the removal is a no-op and we just train
on the same items. Do a one-shot scan first:

```bash
uv run python -c "
import pickle
from collections import Counter
for src in ['benchcad', 'cad-iso-106', 'benchcad-simple', 'text2cad-bench', 'cad-recode-bench']:
    rows = pickle.load(open(f'data/{src}/train.pkl', 'rb'))
    c = Counter(r.get('family') for r in rows)
    print(f'{src}: {len(c)} families, top-10: {c.most_common(10)}')
"
```

Pick the 2–4 actual families that (a) exist in our train pkl, (b) have
≥200 items each so we can hold out a meaningful test subset, and
(c) span complexity (1 simple, 2 medium, 1 hard).

**2. Rare-ops list — align with our existing parser.**
Our `train/sft/online_eval._OPS` and
`scripts/analysis/parse_cq.py` already enumerate the cadquery ops we
detect via regex. The user's list includes `helix`, `thread`,
`gear_tooth_profile`, `twisted_extrude` — none of which are standard
cadquery method names. If the parser doesn't detect them, recall is
trivially zero for everyone. Two options:

  (a) Stick to ops we already parse. From the existing `_OPS` set, the
      genuinely rare ones are: `loft`, `sweep`, `revolve`, `mirror`,
      `shell`, `chamfer` (rare in BenchCAD-only, common in
      cad-recode-bench), `cbore`, `csk`. That's a natural rare set.

  (b) Extend the parser. Add regex for `\.helix\(`, `\.assembly\.…`,
      `\.threads\.…`. Half a day of work, but defensible if reviewers
      ask "did your op set include thread / helix?"

I recommend (a) for speed + reliability; mention (b) as a limitation.

**3. RL gate — be explicit that v3 hasn't crossed it.**
The plan assumes a clean (Base, SFT, RL) ladder. Right now:
  - Base: Qwen3-VL-2B raw
  - SFT: v3 best ckpt (currently at step 43k, ~88%; max@8 DC 0.785,
    greedy DC 0.714)
  - RL: NOT YET TRAINED (gate was originally greedy DC ≥ 0.8; we'll
    end ~0.71-0.72)

Either:
  - Train RL anyway from current v3 ckpt and see if it lifts
    greedy → max@8 (the 0.714 → 0.785 gap is exactly what RL chases).
    Even if RL doesn't reach 0.8, the *trajectory* (Base → SFT → RL)
    still tells the right story.
  - OR drop RL from the paper and run only Base vs SFT — but then we
    lose half the narrative.

Recommend: short RL run (~5-10k steps) from v3 best ckpt, accept it
won't break 0.8 greedy, report the *gap reduction* as the contribution.

**4. Feature Count Accuracy — punt to appendix.**
This needs a per-family parser (gear teeth, handwheel spokes, flange
holes). Building 4-5 robust parsers is non-trivial; if regex breaks on
synth code variations, feature count = wrong even when shape is right.
Recommend: include in plan as future work, only run on 1 family
(gear teeth) for the appendix table, full ablation in the journal
extension.

**5. Compute budget reality check.**
Done correctly, this plan needs:
  - 1 unseen-family retrain (~25 h SFT) — yes
  - Re-eval base+SFT+RL on full + held-out splits — ~2 h × 6 cells
  - Re-implement 4 new metrics in eval pipeline — half day
  - Maybe 1 short RL run — ~12 h
  Total: ~3 days of GPU + a couple days of code work. Plan around this.

**6. What's already done — don't redo.**
  - `Op Recall` / `Advanced Op Recall` (≈ what we call `op_macro_recall`,
    `rare_op_macro_recall`) — already computed every eval, in
    `train/sft/online_eval._multilabel_op_metrics`
  - `Unique Valid Ops` ≈ existing `distinct_ops` field per bucket
  - Per-case predictions JSONL already saved (Priority 0 done)

We mostly need to add:
  - Family-removal training data filter
  - "Valid Rare Op Recall" (= rare op recall × exec_ok mask)
  - The metrics writeup as a clean offline-eval script that reads the
    predictions/ jsonl and produces the result tables.
