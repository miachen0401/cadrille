# Progress

## Status Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked

---

## RL Fine-Tuning Reproduction (see plan.md)

- [ ] `reward.py` — IoU-based reward via safe subprocess execution
- [ ] `mine_hard_examples.py` — pre-filter training data for RL
- [ ] `rl_train.py` — Dr. CPPO + DPO training loop
- [ ] `cadrille.py` — add `compute_sequence_logprob()` static method
- [ ] Smoke test (reward module, hard example mining, CPPO 10-step, eval)
