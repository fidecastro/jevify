# Calibration: deepseek-v4-flash-vision-exp-keys.vllm

Fitted on `runs/deepseek-v4-flash-vision-exp-keys.vllm--policy-29--20260920T013053Z.jsonl` (sha256 `bcb9e1bc8c941baa…`), recipe hash before `00a99fdb3bb059c1…`.

Split: 14 decisions to fit, 15 held out (sha256(case_id)[0] parity).

| bucket | fit decisions | temperature |
|---|---|---|
| choice:3-5 | 12 | 0.109 |
| choice:2 | 2 | 0.299 |

| held-out metric | before | after |
|---|---|---|
| log_loss | 0.0137 | 0.0000 |
| brier | 0.0044 | 0.0000 |
| ece | 0.0125 | 0.0000 |

**Not applicable as fitted:**
- choice:3-5: 12 fit decisions, fewer than 50
- choice:3-5: no wrong decision in the fit split, nothing to calibrate
- choice:2: 2 fit decisions, fewer than 50
- choice:2: no wrong decision in the fit split, nothing to calibrate

a temperature never changes a ranking; judge it on held-out log loss and ECE, and on enough decisions: below a few hundred these numbers are indicative only.
