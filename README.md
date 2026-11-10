# Step-INSPECTOR: Decoding-Free Reasoning Error Localization via Step-Level Representation Probing

[![Course](https://img.shields.io/badge/Course-AI60213%20FLLM-blue.svg)](https://moodle.iitkgp.ac.in)
[![Institution](https://img.shields.io/badge/IIT%20Kharagpur-AI%20Department-orange.svg)](https://ai.iitkgp.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Backbone-Qwen2.5--0.5B--Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

> **Course Project:** Foundation of Large Language Models (AI60213)  
> **Instructor:** Prof. Plaban Kumar Bhowmick  
> **Group Number:** Group [Insert Group Number]

---

## Abstract

Large Language Models (LLMs) are widely adopted as reference-free evaluators, yet prompt-based evaluation remains computationally demanding, prone to verbosity and position bias, and opaque. Recent work, *INSPECTOR* (Li et al., 2026), validates the **Semantic Capacity Asymmetry Hypothesis**: small language models (SLMs) encode rich evaluative signals in their internal representations despite weak generative ability. However, existing representation-as-a-judge approaches evaluate only coarse, whole-response quality, failing to pinpoint where multi-step reasoning breaks down.

In this project, we introduce **Step-INSPECTOR**, an empirical extension that investigates whether frozen SLM internal representations can achieve **decoding-free mathematical error localization**. Using step-annotated candidate solutions from **MR-GSM8K**, we extract intermediate-layer representations of a frozen `Qwen2.5-0.5B-Instruct` model and train lightweight step probes. We introduce a causal masked-loss formulation to prevent label contamination from cascading errors and evaluate layer-depth dynamics, pooling mechanisms, and position/length biases.

## Core Contributions

1. **Sub-solution granularity:** Extends the *Representation-as-a-Judge* paradigm from whole-response scoring to step-level error detection without generating critique text.
2. **Causal supervision masking:** Implements a dynamic loss mask that supervises steps up to the first error index \(k^*\) and masks post-error steps, preventing corrupted deductive chains from injecting noise into the probe.
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

**Google Drive download link:** [Insert your team's Drive link here]

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

  \[
  \mathrm{FirstErrorAccuracy} =
  \frac{\sum_{i: k_i^*>0} \mathbf{1}[\hat{k}_i = k_i^*]}
       {\sum_i \mathbf{1}[k_i^*>0]}.
  \]

- **Joint Accuracy:** Percentage of instances for which both the verdict and the exact first-error step index are correct.

## References and Prior Work

- **INSPECTOR (base paper):** Li et al. (2026). *Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry.* [arXiv:2601.22588](https://arxiv.org/abs/2601.22588)
- **ReProbe:** Ni et al. (ACL 2026). *ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models.*
- **MR-GSM8K:** Zeng et al. (2024). *MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation.*
- **Qwen2.5:** Yang et al. (2024). *Qwen2.5 Technical Report.*

## Academic Integrity and AI Declaration

In accordance with course guidelines for AI60213, generative AI tools were utilized strictly as a conceptual brainstorming aid to formulate the theoretical extension of the INSPECTOR framework and assist in LaTeX formatting. All data preprocessing, token span offset mapping, PyTorch architectures, loss-masking formulations, and experimental evaluations were implemented directly by the group members.
