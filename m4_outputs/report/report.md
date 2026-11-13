# Member 4 - results (auto-generated)

Protocol: fitting, layer choice and thresholds use 5-fold out-of-fold predictions on train+val; the test split is scored once. Intervals are 95% bootstraps over solutions. Numbers are empirical estimates, not causal proof.

- **Qwen2.5-0.5B**: layer 19 of 24 (max out-of-fold step AUROC on train+val), hidden 896, test = 100 solutions (50 faulty).
- **Qwen3-0.6B**: layer 16 of 28 (max out-of-fold step AUROC on train+val), hidden 1024, test = 100 solutions (50 faulty).

## 1. Threshold-free results (primary)

| Model | Method | Step AUROC [95% CI] | AUROC without answer lines | Top-step accuracy % [95% CI] |
|---|---|---|---|---|
| Qwen2.5-0.5B | Step probe | 0.880 [0.84-0.92] | 0.849 | 38.0 [25-51] |
| Qwen2.5-0.5B | Position only | 0.506 [0.42-0.59] | 0.441 | 14.0 [5-24] |
| Qwen2.5-0.5B | Length only | 0.708 [0.65-0.76] | 0.626 | 24.0 [13-36] |
| Qwen2.5-0.5B | Position + length | 0.716 [0.66-0.77] | 0.636 | 22.0 [11-33] |
| Qwen3-0.6B | Step probe | 0.857 [0.80-0.90] | 0.823 | 36.0 [22-50] |
| Qwen3-0.6B | Position only | 0.506 [0.42-0.59] | 0.441 | 14.0 [5-24] |
| Qwen3-0.6B | Length only | 0.708 [0.65-0.76] | 0.626 | 24.0 [13-36] |
| Qwen3-0.6B | Position + length | 0.716 [0.66-0.77] | 0.636 | 22.0 [11-33] |

Top-step accuracy = when the solution is known to be faulty, the highest-scoring step is the gold first-error step.

| Model | Random step (expected) | Modal step from train+val | Always answer 'correct' (joint) |
|---|---|---|---|
| Qwen2.5-0.5B | 13.5 | 10.0 (step 1) | 50.0 |
| Qwen3-0.6B | 13.5 | 10.0 (step 1) | 50.0 |

## 2. Operating point 'balanced' (maximise mean of verdict and first-error on OOF)

| Model | Method | Threshold | Verdict % | First-error % | Joint % |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Step probe | 0.300 | 73.0 [64-81] | 34.0 [21-47] | 55.0 [45-64] |
| Qwen2.5-0.5B | Position only | 0.075 | 50.0 [40-60] | 10.0 [2-19] | 5.0 [1-10] |
| Qwen2.5-0.5B | Length only | 0.150 | 55.0 [45-65] | 16.0 [6-27] | 28.0 [20-37] |
| Qwen2.5-0.5B | Position + length | 0.125 | 51.0 [41-61] | 18.0 [8-29] | 14.0 [7-21] |
| Qwen3-0.6B | Step probe | 0.225 | 71.0 [62-79] | 34.0 [20-47] | 52.0 [42-62] |
| Qwen3-0.6B | Position only | 0.075 | 50.0 [40-60] | 10.0 [2-19] | 5.0 [1-10] |
| Qwen3-0.6B | Length only | 0.150 | 55.0 [45-65] | 16.0 [6-27] | 28.0 [20-37] |
| Qwen3-0.6B | Position + length | 0.125 | 51.0 [41-61] | 18.0 [8-29] | 14.0 [7-21] |

## 3. Operating point 'joint' (maximise joint accuracy on OOF)

| Model | Method | Threshold | Verdict % | First-error % | Joint % |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Step probe | 0.450 | 74.0 [66-82] | 38.0 [25-52] | 61.0 [51-70] |
| Qwen2.5-0.5B | Position only | 0.500 | 50.0 [40-60] | 0.0 [0-0] | 50.0 [40-60] |
| Qwen2.5-0.5B | Length only | 0.550 | 50.0 [40-60] | 0.0 [0-0] | 50.0 [40-60] |
| Qwen2.5-0.5B | Position + length | 0.650 | 50.0 [40-60] | 0.0 [0-0] | 50.0 [40-60] |
| Qwen3-0.6B | Step probe | 0.850 | 68.0 [59-77] | 14.0 [5-24] | 53.0 [43-63] |
| Qwen3-0.6B | Position only | 0.500 | 50.0 [40-60] | 0.0 [0-0] | 50.0 [40-60] |
| Qwen3-0.6B | Length only | 0.550 | 50.0 [40-60] | 0.0 [0-0] | 50.0 [40-60] |
| Qwen3-0.6B | Position + length | 0.650 | 50.0 [40-60] | 0.0 [0-0] | 50.0 [40-60] |

Joint accuracy cannot beat the 'always correct' reference on a 50/50 test set unless first-error accuracy is also high; this is why the balanced operating point is reported next to it.

## 4. Paired difference, probe minus baseline (95% CI; interval excluding 0 = clear difference)

| Model | Baseline | Step AUROC | Top-step acc (pts) | Balanced verdict (pts) | Balanced first-error (pts) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Position only | [+0.286, +0.463] | [+8.0, +39.5] | [+10.0, +36.0] | [+7.4, +40.4] |
| Qwen2.5-0.5B | Length only | [+0.109, +0.238] | [-1.8, +28.9] | [+5.0, +30.0] | [+4.7, +31.5] |
| Qwen2.5-0.5B | Position + length | [+0.103, +0.227] | [+1.7, +30.0] | [+9.0, +35.0] | [+1.9, +31.4] |
| Qwen3-0.6B | Position only | [+0.258, +0.440] | [+6.1, +38.0] | [+8.0, +34.0] | [+9.8, +38.8] |
| Qwen3-0.6B | Length only | [+0.073, +0.222] | [-2.1, +26.5] | [+3.0, +29.0] | [+6.1, +30.4] |
| Qwen3-0.6B | Position + length | [+0.065, +0.214] | [+0.0, +27.5] | [+7.0, +32.0] | [+2.6, +29.5] |

## 5. Length audit of the probe (observational)

| Model | rho(length, score), correct steps [95% CI] | AUROC short | AUROC mid | AUROC long |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.36 [0.25, 0.46] | 0.9463 | 0.8001 | 0.8122 |
| Qwen3-0.6B | 0.37 [0.26, 0.49] | 0.9646 | 0.7659 | 0.7711 |

A positive rho on correct steps means longer steps get higher error scores even when they are correct. Falling AUROC from short to long steps means errors are harder to detect in long steps.

## 6. Padding stress test (paired, same step with and without added text)

| Model | Padding | Group | n | +tokens | Mean change in error score [95% CI] | flagged->unflagged | unflagged->flagged |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | filler | correct | 20 | 21 | +0.149 [+0.091, +0.207] | 0 | 5 |
| Qwen2.5-0.5B | filler | first_error | 20 | 21 | +0.169 [+0.098, +0.246] | 0 | 6 |
| Qwen2.5-0.5B | plain | correct | 20 | 15 | +0.308 [+0.211, +0.412] | 0 | 9 |
| Qwen2.5-0.5B | plain | first_error | 20 | 15 | +0.326 [+0.212, +0.430] | 0 | 7 |
| Qwen2.5-0.5B | repeat | correct | 20 | 32 | +0.363 [+0.222, +0.505] | 0 | 9 |
| Qwen2.5-0.5B | repeat | first_error | 20 | 31 | +0.312 [+0.172, +0.453] | 1 | 8 |
| Qwen3-0.6B | filler | correct | 20 | 21 | +0.091 [+0.035, +0.159] | 0 | 3 |
| Qwen3-0.6B | filler | first_error | 20 | 21 | +0.188 [+0.120, +0.255] | 0 | 8 |
| Qwen3-0.6B | plain | correct | 20 | 15 | +0.083 [+0.035, +0.142] | 0 | 3 |
| Qwen3-0.6B | plain | first_error | 20 | 15 | +0.165 [+0.106, +0.229] | 0 | 4 |
| Qwen3-0.6B | repeat | correct | 20 | 32 | +0.003 [-0.028, +0.034] | 0 | 1 |
| Qwen3-0.6B | repeat | first_error | 20 | 31 | +0.001 [-0.076, +0.073] | 1 | 2 |

Added text changes meaning as well as length: this shows sensitivity to the intervention, not proof of length bias.

## 7. Data audit

- Qwen2.5-0.5B: **PASS**; leakage overlaps {'source_id:train&val': 0, 'source_id:train&test': 0, 'source_id:val&test': 0, 'question:train&val': 0, 'question:train&test': 0, 'question:val&test': 0, 'candidate:train&val': 0, 'candidate:train&test': 0, 'candidate:val&test': 0}; parity: worst cosine 0.999995
- Qwen3-0.6B: **PASS**; leakage overlaps {'source_id:train&val': 0, 'source_id:train&test': 0, 'source_id:val&test': 0, 'question:train&val': 0, 'question:train&test': 0, 'question:val&test': 0, 'candidate:train&val': 0, 'candidate:train&test': 0, 'candidate:val&test': 0}; parity: worst cosine 0.99999

## Limitations

- Only 50 faulty test solutions: intervals are about +/-13 points; do not rank differences smaller than that.
- The 'balanced' operating point was declared after seeing that 'joint' is degenerate on 50/50 data; report both.
- Length findings are observational; the padding test changes wording as well as length.
- Layer choice used out-of-fold AUROC on train+val only; neighbouring layers are usually within noise.
