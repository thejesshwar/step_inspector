# Step-INSPECTOR: Decoding-Free Reasoning Error Localization via Step-Level Representation Probing

[![Course](https://img.shields.io/badge/Course-AI60213%20FLLM-blue.svg)](https://moodle.iitkgp.ac.in)
[![Institution](https://img.shields.io/badge/IIT%20Kharagpur-AI%20Department-orange.svg)](https://ai.iitkgp.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Models](https://img.shields.io/badge/Backbones-Qwen2.5--0.5B%20%7C%20Qwen3--0.6B-purple.svg)](https://huggingface.co/Qwen)

> **Course Project:** Foundation of Large Language Models (AI60213)  
> **Instructors:** Prof. Plaban Kumar Bhowmick and Dr. Shreya Ghosh  
> **Group Number:** Group 6  
> **Team Members:**  
> - **Member 1:** V Rahul Sai Indeevar (24AI10048) — *Spatial & Pooling Dynamics*  
> - **Member 2:** Sai Chandan (24AI10011) — *Core Architecture & Probe Capacity*  
> - **Member 3:** M B Thejesshwar (24AI10035) — *Layer-Depth & Causal Masking Dynamics*  
> - **Member 4:** Ruthvik Devarakonda (24AI10023) — *Diagnostic Benchmarking & Meta-Evaluation*

---

## Abstract

Large Language Models (LLMs) are widely adopted as reference-free evaluators, yet prompt-based evaluation remains computationally demanding, prone to verbosity and position bias, and opaque. Recent work, *INSPECTOR* (Li et al., 2026), validates the **Semantic Capacity Asymmetry Hypothesis**: small language models (SLMs) encode rich evaluative signals in their internal representations despite weak generative ability. However, existing representation-as-a-judge approaches evaluate only coarse, whole-response quality, failing to pinpoint where multi-step reasoning breaks down.

In this project, we introduce **Step-INSPECTOR**, an empirical extension that investigates whether frozen SLM internal representations can achieve **decoding-free mathematical error localization**. Using step-annotated candidate solutions from **MR-GSM8K**, we extract intermediate-layer representations from two frozen backbones—`Qwen2.5-0.5B-Instruct` (24 layers, 896d) and `Qwen3-0.6B` (28 layers, 1024d)—and train lightweight step probes. We introduce a causal masked-loss formulation ($\mathcal{L}_{\text{causal}}$) to prevent label contamination from cascading errors and evaluate layer-depth dynamics, pooling mechanisms, probe capacities, and position/length biases.

## Core Contributions

1. **Sub-solution granularity (Member 1):** Extends the *Representation-as-a-Judge* paradigm from whole-response scoring to step-level error detection using deterministic regex step slicing (verified 650/650 on MR-GSM8K) and tokenizer offset mapping without generating critique text.
2. **Probe capacity & linear separability analysis (Member 2):** Compares parameter-minimal Linear Probes against 2-layer LayerNorm+GELU MLP Probes across multiple seeds and confusion-matrix diagnostics.
3. **Causal supervision masking & layer-depth geometry (Member 3):** Implements a dynamic causal loss mask ($\mathcal{L}_{\text{causal}}$) that supervises steps up to the first error index $k^*$ and masks post-error steps ($m > k^*$), alongside single-layer vs. Cross-Layer Attention Probe (CLAP) comparisons across all 24/28 layers.
4. **Diagnostic and bias auditing (Member 4):** Implements a leakage-free 5-fold Out-of-Fold (OOF) evaluation protocol with solution-clustered bootstrap 95% CIs, comparing Step-INSPECTOR against Position-Only and Length-Only baselines and auditing verbosity sensitivity via controlled padding interventions.

## Repository Structure

```text
step_inspector/
├── data/                         # Cached multi-layer tensors (Qwen3-0.6B) & dataset JSON
├── backup_qwen25/                # Cached multi-layer tensors & plots for Qwen2.5-0.5B
├── results/role2/                # Member 2 multi-seed capacity ablation CSVs (qwen25 & qwen3)
├── m4_outputs/                   # Member 4 OOF layer sweeps, audits, stress tests & figures
├── extract_and_cache.py          # Master GPU feature extraction & causal mask caching script
├── spatial_pooling.py            # Step-span offset mapping & Mean/Last/Global pooling (Member 1)
├── plot_pooling_ablation.py      # Spatial pooling ablation figure & summary generator (Member 1)
├── models.py                     # LinearProbe and MLPProbe architectures (Member 2)
├── run_role2_capacity.py         # Multi-seed probe capacity ablation & diagnostics (Member 2)
├── verify_role2_data.py          # Cross-backbone cache alignment checker (Member 2)
├── custom_loss.py                # Causal Masked Step BCE Loss module (Member 3)
├── run_role3_depth.py            # Layer-depth probing & Cross-Layer Attention Probe (Member 3)
├── run_role3_masking_ablation.py # Causal Masking vs. Naive Unmasked ablation (Member 3)
├── m4_common.py                  # Shared OOF CV, bootstrap & metric utilities (Member 4)
├── m4_evaluate.py                # Main OOF diagnostic benchmark & length tertile audit (Member 4)
├── m4_audit.py                   # Cache integrity, split leakage & extraction parity audit (Member 4)
├── m4_stress_test.py             # Controlled text-padding intervention stress test (Member 4)
├── m4_report.py                  # Automated Markdown/LaTeX/Figure generator (Member 4)
├── check_data.py                 # Dataset split inspection utility
├── verify_steps.py               # 650/650 step-boundary assertion checker
└── README.md
```

## Setup and Installation

```bash
git clone https://github.com/thejesshwar/step_inspector.git
cd step_inspector
python -m venv venv
# Linux/macOS: source venv/bin/activate | Windows: .\venv\Scripts\Activate.ps1
pip install torch transformers accelerate tqdm matplotlib numpy scikit-learn scipy
```

**Pre-computed representation tensors:** Download the cached `.pt` files from Google Drive and place them in `data/` (and `backup_qwen25/`):  
[https://drive.google.com/drive/folders/1pjWxnrtSfe1NMAjgwtM2z0wtdWsVpD-R?usp=sharing](https://drive.google.com/drive/folders/1pjWxnrtSfe1NMAjgwtM2z0wtdWsVpD-R?usp=sharing)

---

## Results & Findings

### Member 1: Spatial & Pooling Dynamics

Implemented in `spatial_pooling.py` and `plot_pooling_ablation.py`.

**Pooling Ablation Summary (Test Split, First-Error Accuracy):**

| Model | Strategy | OOF Layer (L19/L16) FE-Acc (%) | Val Layer (L14/L17) FE-Acc (%) | Peak Layer | Peak FE-Acc (%) | Verdict Acc at Peak (%) | Joint Acc at Peak (%) | All-Layer Mean FE-Acc (%) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (24L) | Mean Pooling | 24.0 | 36.0 | 18 | **40.0** | 67.0 | **50.0** | 23.1 |
| Qwen2.5-0.5B (24L) | Last-Token Pooling | 28.0 | 28.0 | 18 | 36.0 | 66.0 | 42.0 | **27.5** |
| Qwen2.5-0.5B (24L) | Global (Whole-Response) | 2.0 | 2.0 | 12 | 8.0 | 66.0 | 44.0 | 4.4 |
| Qwen3-0.6B (28L) | Mean Pooling | 22.0 | **32.0** | 24 | **38.0** | 68.0 | **48.0** | **23.7** |
| Qwen3-0.6B (28L) | Last-Token Pooling | **26.0** | 16.0 | 21 | 30.0 | 64.0 | 44.0 | 21.6 |
| Qwen3-0.6B (28L) | Global (Whole-Response) | 4.0 | 4.0 | 25 | 8.0 | 66.0 | 47.0 | 5.1 |

*Takeaway:* Global whole-response pooling (base INSPECTOR baseline) collapses on step localization (~4.4%–5.1% mean FE-Acc, well below the 13.5% random step baseline), whereas step-span Mean Pooling achieves the highest peak localization on both models (40.0% on Qwen2.5; 38.0% on Qwen3).

```bash
python spatial_pooling.py
python plot_pooling_ablation.py
```

---

### Member 2: Core Architecture & Probing (Probe Capacity Ablation)

Implemented in `models.py`, `run_role2_capacity.py`, and `verify_role2_data.py`.

**Probe Capacity Ablation Summary (Layer 22, 3 Seeds `{42, 123, 2026}`, Mean ± SD):**

| Backbone | Probe | Params | Verdict Acc (%) | First-Error Acc (%) | Joint Acc (%) | Recall (%) | FPR (%) | F1 (%) |
|---|---|---:|---|---|---|---|---|---|
| Qwen2.5-0.5B | Linear | 897 | 67.33 ± 0.58 | 23.33 ± 3.06 | **60.67 ± 1.53** | 36.67 ± 1.15 | **2.00 ± 0.00** | 52.88 ± 1.22 |
| Qwen2.5-0.5B | MLP | 201,601 | **70.67 ± 1.53** | **32.00 ± 5.29** | 60.00 ± 1.00 | **53.33 ± 7.02** | 12.00 ± 4.00 | **64.33 ± 4.27** |
| Qwen3-0.6B | Linear | 1,025 | 69.33 ± 5.03 | **24.67 ± 7.57** | 58.33 ± 2.52 | 46.67 ± 12.70 | 8.00 ± 5.29 | 59.61 ± 10.99 |
| Qwen3-0.6B | MLP | 263,169 | **71.67 ± 3.21** | 24.00 ± 6.00 | **59.33 ± 2.08** | **48.67 ± 8.08** | **5.33 ± 2.31** | **62.93 ± 6.60** |

*Takeaway:* `MLPProbe` improves First-Error Accuracy on Qwen2.5-0.5B (+8.67 ± 7.02 pp) by raising error recall from 36.67% to 53.33%, but provides no localization gain on Qwen3-0.6B (−0.67 ± 4.62 pp), indicating strong linear separability in Qwen3-0.6B representations.

```bash
python verify_role2_data.py
python run_role2_capacity.py --data-dir backup_qwen25 --output-dir results/role2/qwen25
python run_role2_capacity.py --data-dir data --output-dir results/role2/qwen3
```

---

### Member 3: Layer-Depth & Representation Dynamics (Causal Masking Track)

Implemented in `run_role3_depth.py`, `custom_loss.py`, and `run_role3_masking_ablation.py`.

**1. Layer-Depth & Cross-Layer Attention Findings**
- Evaluative error-detection features at $\tau = 0.5$ are near 0% in shallow layers (Layers 1–11) and emerge sharply in mid-to-late layers (Layers 14–22 on Qwen2.5-0.5B; Layers 15–28 on Qwen3-0.6B).
- Under strict 5-fold OOF selection, **Qwen2.5-0.5B selects Layer 19** (22.0% Test FE-Acc at $\tau=0.5$, 59.0% Joint Acc; peaking at Layer 22 with 36.0% FE-Acc, 60.0% Joint Acc) and **Qwen3-0.6B selects Layer 16** (32.0% Test FE-Acc, 60.0% Joint Acc).
- **Single-Layer vs. Cross-Layer Attention Probe (CLAP):** Single-layer probes outperform CLAP on both Qwen2.5-0.5B (L19: 22.0% FE-Acc / 59.0% Joint; L22: 36.0% FE-Acc / 60.0% Joint vs. CLAP: 20.0% FE-Acc / 52.0% Joint) and Qwen3-0.6B (L16: 32.0% FE-Acc / 60.0% Joint vs. CLAP: 26.0% FE-Acc / 57.0% Joint), as cross-layer attention overfits to noisy shallow layers on 500 training examples.

**2. Causal Loss Masking Ablation**

| Model (Layer) | Supervision Masking Scheme | Verdict Acc (%) | First-Error Acc (%) | Joint Acc (%) |
|---|---|---|---|---|
| **Qwen3-0.6B (OOF L16)** | **Proposed Causal Masking** | 74.0 | **32.0** | **60.0** |
| Qwen3-0.6B (OOF L16) | Naive Unmasked (Target = 0) | 65.0 | 12.0 | 54.0 |
| Qwen3-0.6B (OOF L16) | Naive Unmasked (Target = 1) | **77.0** | 30.0 | 57.0 |
| **Qwen2.5-0.5B (OOF L19)** | **Proposed Causal Masking** | 65.0 | 22.0 | **59.0** |
| Qwen2.5-0.5B (OOF L19) | Naive Unmasked (Target = 0) | 54.0 | 8.0 | 53.0 |
| Qwen2.5-0.5B (OOF L19) | Naive Unmasked (Target = 1) | **71.0** | **26.0** | 56.0 |
| **Qwen2.5-0.5B (Peak L22)** | **Proposed Causal Masking** | **72.0** | **36.0** | **60.0** |
| Qwen2.5-0.5B (Peak L22) | Naive Unmasked (Target = 0) | 62.0 | 18.0 | 54.0 |
| Qwen2.5-0.5B (Peak L22) | Naive Unmasked (Target = 1) | 71.0 | 30.0 | 57.0 |

```bash
python run_role3_depth.py
python run_role3_masking_ablation.py
```

---

### Member 4: Diagnostic Benchmarks & Meta-Evaluation

Implemented in `m4_evaluate.py`, `m4_common.py`, `m4_audit.py`, `m4_stress_test.py`, and `m4_report.py`.

**Main Results (Test Split: 100 solutions, 50 faulty; Balanced Operating Point):**

| Model | Method | Step AUROC [95% CI] | No-Ans AUROC | Top-Step Acc [95% CI] | Verdict [%] | First-Error [%] | Joint [%] |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (L19) | Step probe | **0.880** [0.84–0.92] | **0.849** | **38.0** [25–51] | **73.0** | **34.0** | **55.0** |
| Qwen2.5-0.5B (L19) | Position only | 0.506 [0.42–0.59] | 0.441 | 14.0 [5–24] | 50.0 | 10.0 | 5.0 |
| Qwen2.5-0.5B (L19) | Length only | 0.708 [0.65–0.76] | 0.626 | 24.0 [13–36] | 55.0 | 16.0 | 28.0 |
| Qwen3-0.6B (L16) | Step probe | **0.857** [0.80–0.90] | **0.823** | **36.0** [22–50] | **71.0** | **34.0** | **52.0** |
| Qwen3-0.6B (L16) | Position only | 0.506 [0.42–0.59] | 0.441 | 14.0 [5–24] | 50.0 | 10.0 | 5.0 |
| Qwen3-0.6B (L16) | Length only | 0.708 [0.65–0.76] | 0.626 | 24.0 [13–36] | 55.0 | 16.0 | 28.0 |

```bash
python m4_evaluate.py --cache-dir backup_qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
python m4_evaluate.py --cache-dir data --dataset data/assignment3_data.json --out-dir m4_outputs/qwen3
python m4_audit.py --cache-dir backup_qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25 --parity 20
python m4_stress_test.py --cache-dir backup_qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
python m4_report.py --runs qwen25=m4_outputs/qwen25 qwen3=m4_outputs/qwen3 --out-dir m4_outputs/report
```

## Academic Integrity and AI Declaration

In accordance with course guidelines for AI60213, generative AI tools were utilized strictly as a conceptual brainstorming aid to formulate the theoretical extension of the INSPECTOR framework and assist in LaTeX/Markdown formatting. All data preprocessing, token span offset mapping, PyTorch architectures, loss-masking formulations, and experimental evaluations were implemented directly by the group members.