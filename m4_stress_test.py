"""m4_stress_test.py - controlled padding intervention (length / verbosity sensitivity).

    python m4_stress_test.py --cache-dir data/qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
    python m4_stress_test.py --cache-dir data/qwen3  --dataset data/assignment3_data.json --out-dir m4_outputs/qwen3

Run m4_evaluate.py first (this script reads the selected layer from <out-dir>/results.json).
Needs the Hugging Face model (downloaded automatically; GPU optional).

For every selected step the mathematics is copied verbatim and ONE of three paddings is appended:
  filler  - a neutral explanatory sentence,
  plain   - a different neutral sentence (to separate 'length' from 'this particular wording'),
  repeat  - the step's own text repeated once (adds tokens, adds no new content).
Only genuine arithmetic steps are targeted (answer lines such as '#### 31' are excluded).
Probe = the same fold-ensemble probe as m4_evaluate.py, thresholds from OOF on train+val.
Test labels only decide which steps are targeted (correct vs first-error); nothing is trained on test.

Caveat that must stay in the report: any added sentence changes semantics as well as length,
so a score change is evidence of sensitivity to this added text, not proof of 'length bias'.
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

import m4_common as C

TEMPLATES = {
    "filler": " For clarity, this step uses the same quantities and calculation already written above; "
              "it introduces no additional operation.",
    "plain": " This explanation is written in plain language to make the step easier to read.",
    "repeat": None,  # the step's own body, repeated once
}
HEADING = re.compile(r"^Step\s+\d+\s*[:.\-]\s*", re.IGNORECASE)
SYSTEM_PROMPT = "You solve arithmetic word problems accurately."  # must equal extract_and_cache.py


# ------------------------------------------------------------------------------ text surgery
def padding_for(template, step_text):
    if template == "repeat":
        return " " + HEADING.sub("", step_text.strip(), count=1)
    return TEMPLATES[template]


def insert_padding(candidate, step_index, padding):
    """Append `padding` to exactly one 1-based step and prove nothing else changed."""
    spans = C.parse_steps(candidate)
    if not 1 <= step_index <= len(spans):
        raise ValueError(f"step_index={step_index} but only {len(spans)} steps")
    start, end = spans[step_index - 1]
    original = candidate[start:end]
    trimmed = original.rstrip()
    out = candidate[:start] + trimmed + padding + original[len(trimmed):] + candidate[end:]
    new_spans = C.parse_steps(out)
    if len(new_spans) != len(spans):
        raise AssertionError("padding changed the number of steps")
    for j, (s0, e0) in enumerate(spans):
        old = candidate[s0:e0].strip()
        new = out[new_spans[j][0]:new_spans[j][1]].strip()
        if (j == step_index - 1 and new != old + padding) or (j != step_index - 1 and new != old):
            raise AssertionError(f"unexpected text change in step {j + 1}")
    return out


def select_pairs(examples, gold, per_group, seed):
    """Arithmetic, non-answer steps: a random step of correct solutions, the first-error step of faulty ones."""
    rng = random.Random(seed)
    correct, faulty = [], []
    for i, ex in enumerate(examples):
        cand = ex["candidate"]
        texts = [cand[s:e] for s, e in C.parse_steps(cand)]
        ok = [j for j, t in enumerate(texts, 1) if C.ARITHMETIC.search(t) and not C.ANSWER_LINE.search(t)]
        if gold[i] == 0:
            if ok:
                correct.append((i, rng.choice(ok), "correct"))
        elif gold[i] in ok:
            faulty.append((i, int(gold[i]), "first_error"))
    rng.shuffle(correct)
    rng.shuffle(faulty)
    return correct[:per_group] + faulty[:per_group]


# ------------------------------------------------------------------------------ model side
def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return tok, llm


def extract_step_embeddings(question, candidate, tok, llm, layer, device):
    """Mirror of extract_and_cache.py for ONE layer: same prompt, same offsets, same mean pooling."""
    from spatial_pooling import StepSpanExtractor
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question:\n{question}\n\nCandidate solution:\n{candidate}"}]
    try:  # Qwen3 needs thinking disabled (exactly as in extract_and_cache.py); Qwen2.5 ignores the flag
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    start = prompt.index(candidate) if candidate in prompt else prompt.rfind(candidate)
    if start == -1:
        raise ValueError("candidate text not found in chat prompt")
    extractor = StepSpanExtractor(candidate_text=candidate, offset_mapping=enc["offset_mapping"], candidate_start=start)
    ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden = llm(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[layer][0].float()
        pooled = [hidden[sp.tok_start:sp.tok_end].mean(dim=0).cpu() for sp in extractor]
        lens = [int(sp.num_tokens) for sp in extractor]
    if not pooled:
        raise ValueError("no reasoning steps found")
    return torch.stack(pooled).numpy(), np.asarray(lens)


# ------------------------------------------------------------------------------ summary
def summarize(rows, thresholds, seed=0, B=2000):
    rng = np.random.RandomState(seed)
    out = {}
    for template in sorted({r["template"] for r in rows}):
        out[template] = {}
        for group in ("correct", "first_error"):
            sub = [r for r in rows if r["template"] == template and r["group"] == group]
            if not sub:
                continue
            d = np.array([r["delta_probability"] for r in sub])
            boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)]
            cell = {"n": len(sub),
                    "mean_added_tokens": round(float(np.mean([r["added_tokens"] for r in sub])), 1),
                    "mean_original_probability": round(float(np.mean([r["original_probability"] for r in sub])), 4),
                    "mean_padded_probability": round(float(np.mean([r["padded_probability"] for r in sub])), 4),
                    "mean_delta": round(float(d.mean()), 4),
                    "mean_delta_ci95": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
                    "median_delta": round(float(np.median(d)), 4),
                    "fraction_delta_negative": round(float((d < 0).mean()), 3)}
            for op in thresholds:
                cell[f"{op}_flag_on_to_off"] = int(sum(r[f"{op}_flag_orig"] == 1 and r[f"{op}_flag_pad"] == 0 for r in sub))
                cell[f"{op}_flag_off_to_on"] = int(sum(r[f"{op}_flag_orig"] == 0 and r[f"{op}_flag_pad"] == 1 for r in sub))
                cell[f"{op}_whole_response_prediction_changed"] = int(sum(r[f"{op}_pred_orig"] != r[f"{op}_pred_pad"] for r in sub))
            out[template][group] = cell
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True, help="same folder you gave m4_evaluate.py")
    ap.add_argument("--layer", default="auto", help="'auto' = read from <out-dir>/results.json")
    ap.add_argument("--model", default=None, help="Hugging Face id (default: inferred from tensor shape)")
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()

    out = Path(a.out_dir)
    train, val, test = C.load_splits(a.cache_dir)
    tag, model_id, n_layers, dim = C.identify_model(train)
    model_id = a.model or model_id
    if model_id is None:
        raise SystemExit("cannot infer the model id from the tensor shape; pass --model")
    if a.layer == "auto":
        res = json.loads((out / "results.json").read_text())
        layer = int(res["meta"]["layer"])
    else:
        layer = int(a.layer)

    examples = C.load_dataset_split(a.dataset, "test")
    lt = test["step_token_lengths"].numpy().astype(int)
    gt = test["gold_first_errors"].numpy().astype(int)
    C.check_alignment(examples, lt)

    # same probe protocol as m4_evaluate.py
    X, y, m, g, ln = C.pool_layer(train, val, layer)
    oof, predictors = C.cv_fit(X, y, m, g, C.make_mlp_fitter(epochs=a.epochs), a.folds, tuple(range(a.seeds)))
    thresholds = {obj: C.pick_threshold(oof, g, ln, obj) for obj in C.OBJECTIVES}
    Xt = C.layer_arrays(test, layer)[0]
    cached_prob = C.ensemble_predict(predictors, Xt)
    base = {obj: C.metrics(C.decode(cached_prob, lt, t), gt) for obj, t in thresholds.items()}
    print(f"layer {layer}  thresholds {thresholds}\ncached-embedding baseline on test: {json.dumps(base)}")

    pairs = select_pairs(examples, gt, a.per_group, a.seed)
    if not pairs:
        raise SystemExit("no arithmetic target steps found")
    print(f"{len(pairs)} target steps x {len(TEMPLATES)} paddings; loading {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, llm = load_model(model_id, device)

    rows, review, parity = [], [], []
    for n_done, (idx, target, group) in enumerate(pairs, 1):
        ex = examples[idx]
        emb0, len0 = extract_step_embeddings(ex["question"], ex["candidate"], tok, llm, layer, device)
        cached_len = lt[idx][:len(len0)]
        if not np.array_equal(cached_len, len0):
            raise AssertionError(f"token lengths differ from cache at test row {idx}: cache={cached_len.tolist()} fresh={len0.tolist()}")
        p0 = C.ensemble_predict(predictors, emb0[None])[0]
        parity.append(float(np.abs(p0 - cached_prob[idx][:len(p0)]).max()))
        step_text = ex["candidate"][slice(*C.parse_steps(ex["candidate"])[target - 1])]
        for template in TEMPLATES:
            padded = insert_padding(ex["candidate"], target, padding_for(template, step_text))
            emb1, len1 = extract_step_embeddings(ex["question"], padded, tok, llm, layer, device)
            if len(emb1) != len(emb0) or len1[target - 1] <= len0[target - 1]:
                raise AssertionError(f"padding did not lengthen step {target} of row {idx}")
            p1 = C.ensemble_predict(predictors, emb1[None])[0]
            row = {"test_index": idx, "source_id": ex.get("source_id", str(idx)), "group": group, "template": template,
                   "gold_first_error": int(gt[idx]), "target_step": target,
                   "original_tokens": int(len0[target - 1]), "padded_tokens": int(len1[target - 1]),
                   "added_tokens": int(len1[target - 1] - len0[target - 1]),
                   "original_probability": float(p0[target - 1]), "padded_probability": float(p1[target - 1]),
                   "delta_probability": float(p1[target - 1] - p0[target - 1])}
            for op, thr in thresholds.items():
                row[f"{op}_flag_orig"] = int(p0[target - 1] > thr)
                row[f"{op}_flag_pad"] = int(p1[target - 1] > thr)
                row[f"{op}_pred_orig"] = int(C.decode(p0[None], len0[None], thr)[0])
                row[f"{op}_pred_pad"] = int(C.decode(p1[None], len1[None], thr)[0])
            rows.append(row)
            review.append({"test_index": idx, "template": template, "group": group, "target_step": target,
                           "question": ex["question"], "original": ex["candidate"], "padded": padded})
        print(f"[{n_done}/{len(pairs)}] {group:11s} row {idx} step {target}", flush=True)

    sdir = out / "stress"
    sdir.mkdir(parents=True, exist_ok=True)
    with (sdir / "paired_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (sdir / "human_review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"tag": tag, "model_id": model_id, "layer": layer, "device": device, "seed": a.seed,
               "thresholds_from_oof": thresholds, "cached_baseline_test": base,
               "max_abs_probe_probability_difference_fresh_vs_cache": round(max(parity), 6),
               "n_target_steps": len(pairs), "paddings": {k: (v or "<step text repeated>") for k, v in TEMPLATES.items()},
               "results": summarize(rows, thresholds),
               "interpretation": "Paired score change after adding text to one arithmetic step. Added text also changes "
                                 "semantics, so this shows sensitivity to this intervention, not proof of length-bias causality."}
    (sdir / "stress_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))
    print(f"fresh-vs-cache max |probe prob diff| = {summary['max_abs_probe_probability_difference_fresh_vs_cache']}")
    print(f"saved to {sdir}")


if __name__ == "__main__":
    main()
