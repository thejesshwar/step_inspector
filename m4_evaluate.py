"""m4_evaluate.py - Member 4 benchmark for ONE model's cache.

    python m4_evaluate.py --cache-dir data/qwen25 --dataset data/assignment3_data.json --out-dir m4_outputs/qwen25
    python m4_evaluate.py --cache-dir data/qwen3  --dataset data/assignment3_data.json --out-dir m4_outputs/qwen3

Steps
  1. Layer sweep on train+val with 5-fold OOF (test is never read here). Layer = best OOF step-AUROC.
     Skip it by passing --layer N.
  2. Final run at that layer: probe + 3 content-free baselines (position, length, position+length).
  3. Test scored once. Bootstrap CIs over solutions, paired probe-minus-baseline differences,
     length audit, and two OOF-selected operating points (joint / balanced).
Writes <out-dir>/layer_sweep.json and <out-dir>/results.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import m4_common as C


def parse_layers(spec, n_layers):
    if spec == "all":
        return list(range(1, n_layers + 1))
    layers = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            layers += list(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    bad = [l for l in layers if not 1 <= l <= n_layers]
    if bad:
        raise SystemExit(f"layers {bad} outside 1..{n_layers}")
    return layers


def sweep(train, val, layers, args, fitter):
    rows = []
    for L in layers:
        t0 = time.time()
        X, y, m, g, ln = C.pool_layer(train, val, L)
        oof, _ = C.cv_fit(X, y, m, g, fitter, args.folds, tuple(range(args.sweep_seeds)))
        tm = m > 0  # causal mask == trusted steps
        row = {"layer": L, "oof_step_auroc": round(float(roc_auc_score(y[tm], oof[tm])), 4),
               "oof_top_step_acc": round(float(C.top_step_hits(oof, ln, g)[g > 0].mean()), 4)}
        for obj in C.OBJECTIVES:
            thr = C.pick_threshold(oof, g, ln, obj)
            met = C.metrics(C.decode(oof, ln, thr), g)
            row[f"oof_{obj}"] = {"threshold": thr, **{k: round(v, 4) for k, v in met.items()}}
        rows.append(row)
        print(f"layer {L:02d}  OOF AUROC {row['oof_step_auroc']:.3f}  top-step {row['oof_top_step_acc']:.3f}  "
              f"balanced-FE {row['oof_balanced']['first_error']:.3f}  ({time.time() - t0:.1f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, help="folder with train_cached.pt / val_cached.pt / test_cached.pt")
    ap.add_argument("--dataset", required=True, help="data/assignment3_data.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None, help="label for this model (default: auto from tensor shape)")
    ap.add_argument("--layer", default="auto", help="'auto' (OOF-selected) or a 1-based layer number")
    ap.add_argument("--sweep-layers", default="all", help="e.g. all | 10-28 | 12,16,20")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="CV repeats for the final run")
    ap.add_argument("--sweep-seeds", type=int, default=1, help="CV repeats during the layer sweep")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = C.load_splits(a.cache_dir)
    tag, model_id, n_layers, dim = C.identify_model(train, a.tag)
    print(f"model tag={tag}  id={model_id}  layers={n_layers}  hidden={dim}  "
          f"train/val/test = {len(train['gold_first_errors'])}/{len(val['gold_first_errors'])}/{len(test['gold_first_errors'])}")

    examples = C.load_dataset_split(a.dataset, "test")
    lens_t = test["step_token_lengths"].numpy().astype(int)
    C.check_alignment(examples, lens_t)
    fitter = C.make_mlp_fitter(epochs=a.epochs)

    # 1) layer selection (test is never read)
    sweep_rows = None
    if a.layer == "auto":
        layers = parse_layers(a.sweep_layers, n_layers)
        sweep_rows = sweep(train, val, layers, a, fitter)
        best = max(sweep_rows, key=lambda r: (r["oof_step_auroc"], -r["layer"]))
        layer, selected_by = best["layer"], "max out-of-fold step AUROC on train+val"
        (out / "layer_sweep.json").write_text(json.dumps(
            {"tag": tag, "num_layers": n_layers, "hidden_dim": dim, "selected_layer": layer,
             "selected_by": selected_by, "rows": sweep_rows}, indent=2))
        print(f"-> selected layer {layer} ({selected_by})")
    else:
        layer, selected_by = int(a.layer), "fixed by user"
        if not 1 <= layer <= n_layers:
            raise SystemExit(f"--layer must be 1..{n_layers}")

    # 2) final run
    X, y, m, g, ln = C.pool_layer(train, val, layer)
    Xt, yt, mt, gt, lt = C.layer_arrays(test, layer)
    trusted_t = C.trusted_mask(lt, gt)
    answers_t = C.answer_line_mask(examples, lt.shape)
    seeds = tuple(range(a.seeds))
    feats_p, feats_t = C.baseline_features(ln), C.baseline_features(lt)

    methods, ops_all, oof_metrics = {}, {}, {}
    for name in C.METHODS:
        Xp, Xe = (X, Xt) if name == "probe" else (feats_p[name], feats_t[name])
        fit = fitter if name == "probe" else C.lr_fitter
        oof, preds = C.cv_fit(Xp, y, m, g, fit, a.folds, seeds)
        prob = C.ensemble_predict(preds, Xe)
        ops = {obj: C.pick_threshold(oof, g, ln, obj) for obj in C.OBJECTIVES}
        ops_all[name] = ops
        oof_metrics[name] = {obj: C.metrics(C.decode(oof, ln, t), g) for obj, t in ops.items()}
        methods[name] = C.method_summary(prob, oof, ops, gt, lt, yt, trusted_t)
        methods[name]["prob"] = prob
        print(f"{name:16s} done  test AUROC {roc_auc_score(yt[trusted_t], prob[trusted_t]):.3f}", flush=True)

    point, ci, paired = C.bootstrap_all(methods, gt, B=a.boot)

    res_methods = {}
    for name, mth in methods.items():
        prob = mth["prob"]
        keep = trusted_t & ~answers_t
        res_methods[name] = {
            "operating_points": {obj: {"threshold": ops_all[name][obj],
                                       "test": {k: round(point[name][f"{obj}/{k}"], 4) for k in ("verdict", "first_error", "joint")},
                                       "oof": {k: round(v, 4) for k, v in oof_metrics[name][obj].items()}}
                                 for obj in C.OBJECTIVES},
            "step_auroc": round(point[name]["step_auroc"], 4),
            "step_auroc_no_answer_lines": (None if C.safe_auroc(yt[keep], prob[keep]) is None
                                           else round(C.safe_auroc(yt[keep], prob[keep]), 4)),
            "top_step_acc": round(point[name]["top_step_acc"], 4)}

    faulty = gt > 0
    n_steps = (lt > 0).sum(1)
    modal = int(np.bincount(g[g > 0]).argmax())
    reference = {"modal_first_error_step_from_train_val": modal,
                 "modal_step_top_step_acc_test": round(float((gt[faulty] == modal).mean()), 4),
                 "random_step_expected_top_step_acc_test": round(float((1.0 / n_steps[faulty]).mean()), 4),
                 "always_correct_joint_acc_test": round(float((gt == 0).mean()), 4)}

    results = {
        "meta": {"tag": tag, "model_id": model_id, "num_layers": n_layers, "hidden_dim": dim, "layer": layer,
                 "layer_selected_by": selected_by, "n_train": len(train["gold_first_errors"]),
                 "n_val": len(val["gold_first_errors"]), "n_test": len(gt), "n_test_faulty": int(faulty.sum()),
                 "folds": a.folds, "seeds": a.seeds, "epochs": a.epochs, "bootstrap_resamples": a.boot,
                 "torch": torch.__version__},
        "reference": reference,
        "methods": res_methods,
        "ci95": ci,
        "paired_probe_minus_baseline_ci95": paired,
        "length_audit_probe": C.length_audit(lt, gt, yt, methods["probe"]["prob"], answers_t),
        "protocol": ("Fit, layer choice and thresholds use 5-fold OOF on train+val only; test scored once; "
                     "CIs are clustered bootstraps over solutions; 'balanced' operating point was declared "
                     "before the final run because 'joint' is degenerate on a 50/50 test set."),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== TEST RESULTS (layer %d) ===" % layer)
    print(f"{'method':16s} {'AUROC':>6s} {'top-step':>9s} | balanced: verdict  first-err  joint | joint-opt: verdict  first-err  joint")
    for name in C.METHODS:
        r = res_methods[name]
        b, j = r["operating_points"]["balanced"]["test"], r["operating_points"]["joint"]["test"]
        print(f"{name:16s} {r['step_auroc']:6.3f} {r['top_step_acc']:9.3f} | {b['verdict']:15.3f} {b['first_error']:10.3f} {b['joint']:6.3f} "
              f"| {j['verdict']:16.3f} {j['first_error']:10.3f} {j['joint']:6.3f}")
    print("references:", reference)
    print(f"saved {out / 'results.json'}")


if __name__ == "__main__":
    main()
