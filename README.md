# Step-INSPECTOR: Decoding-Free Reasoning Error Localization via Step-Level Representation Probing

[![Course](https://img.shields.io/badge/Course-AI60213%20FLLM-blue.svg)](https://moodle.iitkgp.ac.in)
[![Institution](https://img.shields.io/badge/IIT%20Kharagpur-AI%20Department-orange.svg)](https://ai.iitkgp.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Backbone-Qwen2.5--0.5B--Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

> **Course Project:** Foundation of Large Language Models (AI60213)  
> **Instructor:** Prof. Plaban Kumar Bhowmick and Dr. Shreya Ghosh  
> **Group Number:** Group 6

---

## Abstract

Large Language Models (LLMs) are widely adopted as reference-free evaluators, yet prompt-based evaluation remains computationally demanding, prone to verbosity and position bias, and opaque. Recent work, *INSPECTOR* (Li et al., 2026), validates the **Semantic Capacity Asymmetry Hypothesis**: small language models (SLMs) encode rich evaluative signals in their internal representations despite weak generative ability. However, existing representation-as-a-judge approaches evaluate only coarse, whole-response quality, failing to pinpoint where multi-step reasoning breaks down.

In this project, we introduce **Step-INSPECTOR**, an empirical extension that investigates whether frozen SLM internal representations can achieve **decoding-free mathematical error localization**. Using step-annotated candidate solutions from **MR-GSM8K**, we extract intermediate-layer representations of a frozen `Qwen2.5-0.5B-Instruct` model and train lightweight step probes. We introduce a causal masked-loss formulation to prevent label contamination from cascading errors and evaluate layer-depth dynamics, pooling mechanisms, and position/length biases.

## Core Contributions

1. **Sub-solution granularity:** Extends the *Representation-as-a-Judge* paradigm from whole-response scoring to step-level error detection without generating critique text.
2. **Causal supervision masking:** Implements a dynamic loss mask that supervises steps up to the first error index $k^*$ and masks post-error steps, preventing corrupted deductive chains from injecting noise into the probe.
3. **Layer-depth geometry analysis:** Investigates the residual stream across all 24 layers of Qwen2.5-0.5B to identify the depth at which deductive fallacies become linearly separable.
4. **Diagnostic and bias auditing:** Implements a Position-Only baseline and Length-Bias stress test to assess whether the probe learns semantic deductive validity rather than positional or verbosity shortcuts. The project distinguishes this task setting from self-verification search methods such as *ReProbe* (ACL 2026).

## Team & Vertical Work Allocation

| Member | Track | Key Deliverable | Slide Ownership |
| :--- | :--- | :--- | :---: |
| **Member 1** | Spatial & Pooling Dynamics | Tokenizer offset mapping, Mean vs. Last-token pooling, Base INSPECTOR whole-response baseline | **Slide 2** (Data & Pipeline) |
| **Member 2** | Core Architecture & Probing | PyTorch `StepInspector` module, probe capacity ablation (Linear vs. MLP) | **Slide 4** (Contribution Code) |
| **Member 3** | Layer-Depth & Representation Dynamics | Multi-layer residual extraction, layer-depth progression curve, Cross-Layer Attention probe | **Slide 3** (Architecture) |
| **Member 4** | Diagnostic Benchmarks & Meta-Evaluation | Position-only baseline, Length-bias audit, Joint/First-Error metrics, ICLR report lead | **Slides 5 & 6** (Results & Discussion) |

## Repository Structure

```text
step_inspector/
├── data/
│   ├── assignment3_data.json  # Raw dataset (MR-GSM8K)
│   ├── train_cached.pt       # Pre-computed multi-layer tensors (download separately)
│   ├── val_cached.pt         # Validation cached tensors
│   └── test_cached.pt        # Test cached tensors
├── extract_and_cache.py      # Master GPU feature extraction script
├── run_role3_depth.py        # Layer-depth probing & Cross-Layer Attention (Member 3)
├── models.py                 # PyTorch architectures (StepInspector, LayerDepthInspector)
├── evaluate.py               # Diagnostic baselines & metrics (Member 4)
├── check_data.py             # Data inspection utility
├── verify_steps.py           # 650/650 step-boundary assertion checker
└── README.md
```

## Setup and Installation

### 1. Clone the repository and set up the environment

```bash
git clone https://github.com/thejesshwar/step_inspector.git
cd step_inspector

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install torch transformers accelerate tqdm matplotlib numpy scikit-learn
```

### 2. Download pre-computed representation tensors

The cached hidden states exceed GitHub's 100 MB file limit (approximately 360 MB total). Download the `.pt` files from the team's shared Google Drive and place them in `data/`.

**Google Drive download link:** [https://drive.google.com/drive/folders/1pjWxnrtSfe1NMAjgwtM2z0wtdWsVpD-R?usp=sharing]

Required files:

- `data/train_cached.pt`
- `data/val_cached.pt`
- `data/test_cached.pt`

## Reproduction and Execution

### Option A: Run with pre-cached tensors (CPU/GPU friendly)

Once the `.pt` files are in `data/`, training a probe or running an ablation takes under 15 seconds, according to the project setup.

```bash
# Run Layer-Depth Dynamics Probing (Member 3)
python run_role3_depth.py
```

### Option B: Re-run full extraction from scratch (requires a GPU)

To re-extract representations using `Qwen/Qwen2.5-0.5B-Instruct`:

```bash
python extract_and_cache.py
```

## Evaluation Protocol and Metrics

The evaluation follows the definitions in Assignment 3 and MR-GSM8K.

- **Verdict Accuracy:** Accuracy in distinguishing correct from incorrect solutions.
- **First-Error Accuracy:** Exact step-localization accuracy, evaluated only on ground-truth faulty reasoning chains:

```math
\mathrm{FirstErrorAccuracy}
=
\frac{
    \sum_{i: k_{i}^{*} > 0}
    \mathbf{1}[\hat{k}_{i} = k_{i}^{*}]
}{
    \sum_{i}
    \mathbf{1}[k_{i}^{*} > 0]
}
```

- **Joint Accuracy:** Percentage of instances for which both the verdict and the exact first-error step index are correct.

## References and Prior Work

- **INSPECTOR (base paper):** Li et al. (2026). *Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry.* [arXiv:2601.22588](https://arxiv.org/abs/2601.22588)
- **ReProbe:** Ni et al. (ACL 2026). *ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models.*
- **MR-GSM8K:** Zeng et al. (2024). *MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation.*
- **Qwen2.5:** Yang et al. (2024). *Qwen2.5 Technical Report.*

## Results & Findings

### Member 1: Spatial & Pooling Dynamics

Implemented in `spatial_pooling.py`.

**Setup**
- Step-span slicing via regex (verified 650/650 on MR-GSM8K)
- Tokenizer offset mapping (`return_offsets_mapping=True`) → char spans → token indices
- Cached multi-layer representations from frozen `Qwen2.5-0.5B-Instruct` (24 layers, 896-dim) in `data/`
- Cached multi-layer representations from frozen `Qwen3-0.6B` (28 layers, 1024-dim) in `qwen3/`

**Pooling operators compared**

| Operator | Formula |
|---|---|
| Mean pooling | $h_m = \frac{1}{\lvert s_m \rvert} \sum_{t \in s_m} h_t$ |
| Last-token pooling | $h_m = h_{\text{end}(m)}$ |
| Global (baseline) | Whole-response mean pooling (base INSPECTOR baseline) |

**Finding**
Mean pooling consistently outperforms last-token and global pooling across all layers on both models, confirming that step-level error signals are **distributed across semantic tokens** rather than concentrated at step delimiters — supporting the hypothesis that sub-solution granularity improves error detectability over whole-response pooling.

**Pooling Ablation Summary (Test Split, First-Error Accuracy):**

| Model | Strategy | Best Layer | Best FE-Acc (%) | Verdict Acc (%) | Joint Acc (%) | Mean FE-Acc (%) |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (24L) | Mean Pooling | 18 | 40.0 | 67.0 | 50.0 | 22.8 |
| Qwen2.5-0.5B (24L) | Last-Token Pooling | 18 | 36.0 | 66.0 | 42.0 | 27.5 |
| Qwen2.5-0.5B (24L) | Global (Whole-Response) | 15 | 8.0 | 66.0 | 45.0 | 4.5 |
| Qwen3-0.6B (28L) | Mean Pooling | 24 | 38.0 | 68.0 | 48.0 | 23.7 |
| Qwen3-0.6B (28L) | Last-Token Pooling | 21 | 30.0 | 64.0 | 44.0 | 21.6 |
| Qwen3-0.6B (28L) | Global (Whole-Response) | 25 | 8.0 | 66.0 | 47.0 | 5.1 |

On both models, Mean Pooling achieves the highest First-Error Accuracy (Qwen2.5: 40.0% at Layer 18; Qwen3: 38.0% at Layer 24), while Global (whole-response) pooling stays near chance (~8%) — confirming that whole-response averaging washes out localized step-level errors.

Run the ablation locally:
```bash
python spatial_pooling.py   # Qwen2.5 caches in data/
python plot_pooling_ablation.py  # auto-detects model from cache shape
```

### Member 2: Core Architecture & Probing

*Pending — to be filled by Member 2 (StepInspector module, probe capacity ablation: Linear vs. MLP).*

### Member 3: Layer-Depth & Representation Dynamics (Causal Masking Track)

Implemented in `run_role3_depth.py`, `custom_loss.py`, and `run_role3_masking_ablation.py`.

**Setup**
- Probed all 24 Transformer layers of frozen `Qwen2.5-0.5B-Instruct` using a 2-layer MLP (`StepProbe`: Linear(896, 224) → LayerNorm → GELU → Linear(224, 1)).
- Implemented and evaluated a parametric **Cross-Layer Attention Probe (CLAP)** utilizing layer positional embeddings and scaled dot-product attention across all 24 layers.
- Formulated and ablated the **Causal Masked Step BCE Loss** ($\mathcal{L}_{\text{causal}}$) to isolate true deductive breakdown from post-error premise contamination.

**1. Layer-Depth Dynamics Findings**
- Evaluative error-detection features are virtually absent in shallow layers (Layers 1–2: 0.0% FE-Acc).
- Signal emerges in mid-layers (Layer 15–16: ~24–26% FE-Acc) and peaks sharply at **Layer 22 (36.0% First-Error Accuracy, 73.0% Verdict Accuracy, 63.0% Joint Accuracy)**.
- **Single-Layer vs. Cross-Layer Attention:** The single-layer probe at Layer 22 (36.0%) outperforms the Cross-Layer Attention Probe (22.0%). Due to limited training supervision (500 samples), CLAP's attention weights overfit and attend to uninformative shallow/mid-layer residual noise, whereas the Layer 22 residual stream already concentrates the linear separability of mathematical validity.

**2. Causal Loss Masking Ablation (Layer 22)**

| Supervision Masking Scheme | Verdict Acc (%) | First-Error Acc (%) | Joint Acc (%) |
|---|---|---|---|
| **Proposed Causal Masking** | **72.0** | **36.0** | **60.0** |
| Naive Unmasked (Target = 0) | 62.0 | 18.0 | 54.0 |
| Naive Unmasked (Target = 1) | 71.0 | 30.0 | 57.0 |

*Takeaway:* Causal masking doubles First-Error Accuracy over Naive Unmasked Target=0 (36.0% vs. 18.0%). Downstream steps following an arithmetic failure operate on corrupted premises; unmasked training injects contradictory gradients (penalizing locally sound calculations or penalizing premise sensitivity), whereas causal truncation isolates the exact point of logical failure.

Run the ablation scripts locally:
```bash
python run_role3_depth.py
python run_role3_masking_ablation.py
```
### Member 4: Diagnostic Benchmarks & Meta-Evaluation

Implemented in `m4_evaluate.py`, `m4_common.py`, `m4_audit.py`, `m4_stress_test.py`.

**Protocol**
- Layer & threshold selection via 5-fold out-of-fold on train+val only; test scored once
- Bootstrap 95% CIs clustered over solutions (2000 resamples)
- Primary metrics: step AUROC (threshold-free) and top-step accuracy

**Main Results (Test Split: 100 solutions, 50 faulty)**

| Model | Method | Step AUROC [95% CI] | Top-Step Acc [95% CI] | Verdict [%] | First-Error [%] | Joint [%] |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Step probe | 0.880 [0.84–0.92] | 38.0 [25–51] | 73.0 | 34.0 | 55.0 |
| Qwen2.5-0.5B | Position only | 0.506 [0.42–0.59] | 14.0 [5–24] | 50.0 | 10.0 | 5.0 |
| Qwen2.5-0.5B | Length only | 0.708 [0.65–0.76] | 24.0 [13–36] | 55.0 | 16.0 | 28.0 |
| Qwen3-0.6B | Step probe | 0.857 [0.80–0.90] | 36.0 [22–50] | 71.0 | 34.0 | 52.0 |
| Qwen3-0.6B | Position only | 0.506 [0.42–0.59] | 14.0 [5–24] | 50.0 | 10.0 | 5.0 |
| Qwen3-0.6B | Length only | 0.708 [0.65–0.76] | 24.0 [13–36] | 55.0 | 16.0 | 28.0 |

**Key Takeaways**
- Step probe AUROC (0.86–0.88) clearly exceeds position-only (0.51) and meaningfully exceeds length-only (0.71)
- Without trivial answer lines: probe AUROC 0.849 / 0.823 vs length-only 0.63 — signal is not just answer-line detection
- First-error localization: 38% / 36% vs 13.5% random — but gain over length-only includes zero, so cannot claim probe beats length at locating the error
- Joint accuracy (55% / 52%) is not above the trivial 50% "always correct" baseline on a 50/50 test set
- No winner between models: AUROC gap is 0.023, and out-of-fold ordering reverses on test

**Limitations**
- Only 50 faulty test solutions → intervals ~±13 points; do not rank differences smaller than that
- Length findings are observational; the padding test changes wording as well as length
- "Balanced" operating point was declared after seeing that "joint" is degenerate on 50/50 data; report both

Run the evaluation locally:
```bash
python m4_evaluate.py --cache-dir data/qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
python m4_evaluate.py --cache-dir qwen3 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen3
python m4_audit.py --cache-dir data/qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25 --parity 20
python m4_stress_test.py --cache-dir data/qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
```

## Academic Integrity and AI Declaration

In accordance with course guidelines for AI60213, generative AI tools were utilized strictly as a conceptual brainstorming aid to formulate the theoretical extension of the INSPECTOR framework and assist in LaTeX formatting. All data preprocessing, token span offset mapping, PyTorch architectures, loss-masking formulations, and experimental evaluations were implemented directly by the group members.
