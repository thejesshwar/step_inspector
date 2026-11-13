"""m4_audit.py - integrity, leakage and (optional) extraction-parity audit of one model's cache.

    python m4_audit.py --cache-dir data/qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
    python m4_audit.py --cache-dir data/qwen3  --dataset data/assignment3_data.json --out-dir m4_outputs/qwen3 --parity 20

Without --parity it only reads the cache and dataset (seconds, CPU).
With --parity N it re-runs the frozen model on N test solutions and compares the mean-pooled
step embeddings of EVERY layer with the cache (this is the check that proves the cache matches
the extraction pipeline; 4 examples is too few, use >= 20).
Exit code 1 if any hard check FAILs. Leakage findings are WARN (they are the team's call).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import m4_common as C


def load_all_rows(path):
    root = json.loads(Path(path).read_text(encoding="utf-8"))["splits"]
    return root["train"], root["validation"], root["test"]


def gold_of(ex):
    return 0 if (ex.get("first_error") is None or ex.get("verdict") == "CORRECT") else int(ex["first_error"])


def check_split(name, cache, rows, report):
    gold = cache["gold_first_errors"].numpy().astype(int)
    lens = cache["step_token_lengths"].numpy().astype(int)
    tgt = cache["step_targets"].numpy()
    msk = cache["step_masks"].numpy()
    n, M = lens.shape
    problems = []
    if len(rows) != n:
        problems.append(f"dataset rows {len(rows)} != cache rows {n}")
    else:
        for i, ex in enumerate(rows):
            ns = int((lens[i] > 0).sum())
            k = gold[i]
            if gold_of(ex) != k:
                problems.append(f"row {i}: gold {k} != dataset label {gold_of(ex)}")
            if ns != ex["num_steps"] or ns != len(C.parse_steps(ex["candidate"])):
                problems.append(f"row {i}: step count cache={ns} num_steps={ex['num_steps']} parsed={len(C.parse_steps(ex['candidate']))}")
            if not 0 <= k <= ns:
                problems.append(f"row {i}: gold {k} outside 0..{ns}")
            want_mask = np.zeros(M)
            want_mask[:(ns if k == 0 else k)] = 1
            want_tgt = np.zeros(M)
            if k > 0:
                want_tgt[k - 1] = 1
            if not np.array_equal(msk[i], want_mask):
                problems.append(f"row {i}: causal mask does not match gold first error")
            if not np.array_equal(tgt[i], want_tgt):
                problems.append(f"row {i}: targets do not match gold first error")
    emb = cache["step_embeddings_mean"]
    bad_pad = bad_fin = 0
    pad = torch.from_numpy(lens <= 0)  # [n, M]
    for L in range(emb.shape[1]):
        e = emb[:, L].float()
        bad_fin += int((~torch.isfinite(e)).sum())
        bad_pad += int((e.abs().sum(-1)[pad] > 0).sum())
    if bad_fin:
        problems.append(f"{bad_fin} non-finite embedding values")
    if bad_pad:
        problems.append(f"{bad_pad} padded step slots hold non-zero embeddings")
    report["splits"][name] = {"n": n, "faulty": int((gold > 0).sum()), "correct": int((gold == 0).sum()),
                              "steps_total": int((lens > 0).sum()), "max_steps": int(M),
                              "first_error_histogram": {int(k): int(v) for k, v in sorted(Counter(gold[gold > 0]).items())},
                              "problems": problems[:10], "n_problems": len(problems)}
    return not problems


def leakage(train_rows, val_rows, test_rows):
    out = {}
    sets = {"train": train_rows, "val": val_rows, "test": test_rows}
    for key in ("source_id", "question", "candidate"):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            sa = {r.get(key) for r in sets[a] if r.get(key) is not None}
            sb = {r.get(key) for r in sets[b] if r.get(key) is not None}
            out[f"{key}:{a}&{b}"] = len(sa & sb)
    return out


def parity(cache, rows, model_id, n, layers, report):
    from spatial_pooling import StepSpanExtractor
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    emb, lens = cache["step_embeddings_mean"], cache["step_token_lengths"].numpy().astype(int)
    pick = np.random.RandomState(0).choice(len(rows), size=min(n, len(rows)), replace=False)
    cos_min = np.ones(layers)
    max_abs = np.zeros(layers)
    len_ok = True
    for i in pick:
        ex = rows[i]
        messages = [{"role": "system", "content": "You solve arithmetic word problems accurately."},
                    {"role": "user", "content": f"Question:\n{ex['question']}\n\nCandidate solution:\n{ex['candidate']}"}]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
        start = prompt.index(ex["candidate"]) if ex["candidate"] in prompt else prompt.rfind(ex["candidate"])
        ext = StepSpanExtractor(candidate_text=ex["candidate"], offset_mapping=enc["offset_mapping"], candidate_start=start)
        ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
        with torch.inference_mode():
            hs = torch.stack(llm(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[1:], 0)[:, 0].float()
            for j, sp in enumerate(ext):
                fresh = hs[:, sp.tok_start:sp.tok_end].mean(1).cpu()          # [L, D]
                cached = emb[i, :, j].float()                                  # [L, D]
                cos = torch.nn.functional.cosine_similarity(fresh, cached, dim=-1).numpy()
                cos_min = np.minimum(cos_min, cos)
                max_abs = np.maximum(max_abs, (fresh - cached).abs().max(-1).values.numpy())
                len_ok &= int(sp.num_tokens) == int(lens[i, j])
    report["parity"] = {"model": model_id, "device": device, "n_solutions": int(len(pick)),
                        "token_lengths_match_cache": bool(len_ok),
                        "min_cosine_per_layer": [round(float(c), 6) for c in cos_min],
                        "worst_cosine": round(float(cos_min.min()), 6),
                        "max_abs_diff_per_layer": [round(float(v), 4) for v in max_abs],
                        "note": "cache is fp16; small differences are expected, cosine >= 0.999 is treated as a pass"}
    return bool(len_ok and cos_min.min() >= 0.999)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--parity", type=int, default=0, help="re-extract N test solutions and compare with the cache")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = C.load_splits(a.cache_dir)
    tag, model_id, n_layers, dim = C.identify_model(train)
    model_id = a.model or model_id
    tr_rows, va_rows, te_rows = load_all_rows(a.dataset)
    report = {"tag": tag, "model_id": model_id, "num_layers": n_layers, "hidden_dim": dim, "splits": {}, "checks": {}}

    shapes = {n: tuple(c["step_embeddings_mean"].shape[1:]) for n, c in (("train", train), ("val", val), ("test", test))}
    report["checks"]["shapes_identical_across_splits"] = len(set(shapes.values())) == 1
    report["checks"]["embedding_dtype_fp16"] = str(train["step_embeddings_mean"].dtype) == "torch.float16"
    for name, cache, rows in (("train", train, tr_rows), ("val", val, va_rows), ("test", test, te_rows)):
        report["checks"][f"{name}_labels_masks_embeddings_consistent"] = check_split(name, cache, rows, report)
    report["leakage_overlaps"] = leakage(tr_rows, va_rows, te_rows)
    report["leakage_warning"] = any(v > 0 for v in report["leakage_overlaps"].values())
    if a.parity:
        if model_id is None:
            raise SystemExit("cannot infer model id; pass --model")
        report["checks"]["extraction_parity_with_cache"] = parity(test, te_rows, model_id, a.parity, n_layers, report)
    else:
        report["parity"] = "NOT RUN (use --parity 20)"

    hard = all(report["checks"].values())
    report["status"] = "PASS" if hard else "FAIL"
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("parity",)}, indent=2))
    if a.parity:
        print("parity:", {k: report["parity"][k] for k in ("n_solutions", "worst_cosine", "token_lengths_match_cache")})
    print(f"\nAUDIT {report['status']}" + ("  (WARN: some question/candidate/source_id overlap across splits - read leakage_overlaps)" if report["leakage_warning"] else ""))
    sys.exit(0 if hard else 1)


if __name__ == "__main__":
    main()
