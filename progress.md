# Progress

## Status Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked

---

## Repo Restructuring + Official W&B Logging (2026-03-03)

- [x] Step 0: Commit pre-refactor state (commit 28667f9)
- [x] Step 1: `rl/__init__.py` — empty package marker
- [x] Step 1: `rl/reward.py` — moved from reward.py + `compute_metrics(→ iou_reward, cd)` using scipy cKDTree (8192 points, matches evaluate.py)
- [x] Step 1: `rl/mine.py` — moved from mine_hard_examples.py, updated imports
- [x] Step 2: `rl/train.py` — moved from rl_train.py + official W&B key names:
  - `loss`, `average_reward` at top level (matches official dashboard)
  - `eval/pc/DeepCAD test/IoU mean|median|CD mean|median|Failures fraction`
  - `run_validation()` now calls `compute_metrics()` → IoU + CD per sample
  - log.txt updated to match official key format
- [x] Step 3: `configs/sft/{default,full,h100,smoke}.yaml`
- [x] Step 3: `configs/rl/default.yaml` (+ val_split_dir, val_samples=200, eval_steps=500)
- [x] Step 3: `configs/rl/4080.yaml` (NEW — G=4, Adam8bit, for 16 GB RTX 4080)
- [x] Step 4: `scripts/run_sft.sh` — config paths → `configs/sft/`
- [x] Step 4: `scripts/run_rl.sh` — `rl_train.py` → `rl/train.py`, config paths → `configs/rl/`
- [x] Step 4: `scripts/run_eval.sh` — NEW: one-shot eval pipeline (test.py → evaluate.py)
- [x] Step 5: `requirements.txt` — pinned deps
- [x] Step 6: `README.md` — rewrite with FAIR-style sections + RL training docs

### Old flat files (keep until PID 10126 finishes)
- [ ] Delete `rl_train.py` after running job finishes
- [ ] Delete `reward.py` after running job finishes
- [ ] Delete `mine_hard_examples.py` after running job finishes
- [ ] Delete flat `configs/*.yaml` after migration confirmed

---

## RL Fine-Tuning Reproduction

- [x] `reward.py` — IoU-based reward via safe subprocess execution
- [x] `mine_hard_examples.py` — pre-filter training data for RL
- [x] `rl_train.py` — Dr. CPPO + DPO training loop
- [x] `cadrille.py` — `compute_sequence_logprob()` static method
- [ ] Smoke test (reward module, hard example mining, CPPO 10-step, eval)
