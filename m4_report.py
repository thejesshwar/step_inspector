"""m4_report.py - turn Member 4 outputs of one or more models into report tables and figures.

    python m4_report.py --runs qwen25=m4_outputs/qwen25 qwen3=m4_outputs/qwen3 --out-dir m4_outputs/report

Reads <run>/results.json (required) and, if present, layer_sweep.json, stress/stress_summary.json, audit.json.
Writes report.md, main_results.csv, table_main.tex, fig_layer_sweep.png, fig_main.png, fig_length.png.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed colour per entity (validated categorical slots). Methods use slots 1-4, models use slots 6-8.
METHOD_COLOR = {"probe": "#2a78d6", "position_only": "#eb6834", "length_only": "#1baf7a", "position+length": "#eda100"}
METHOD_LABEL = {"probe": "Step probe", "position_only": "Position only", "length_only": "Length only",
                "position+length": "Position + length"}
MODEL_COLORS = ["#008300", "#6250d6", "#e34948"]
MODEL_MARKERS = ["o", "s", "D"]
NICE = {"qwen25": "Qwen2.5-0.5B", "qwen3": "Qwen3-0.6B"}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def pct_ci(x, ci):
    return f"{pct(x)} [{100 * ci[0]:.0f}-{100 * ci[1]:.0f}]"


def dec_ci(x, ci):
    return f"{x:.3f} [{ci[0]:.2f}-{ci[1]:.2f}]"


def load_runs(specs):
    runs = {}
    for spec in specs:
        name, folder = spec.split("=", 1)
        d = Path(folder)
        runs[name] = {"dir": d, "res": json.loads((d / "results.json").read_text())}
        for key, rel in (("sweep", "layer_sweep.json"), ("stress", "stress/stress_summary.json"), ("audit", "audit.json")):
            p = d / rel
            runs[name][key] = json.loads(p.read_text()) if p.exists() else None
    return runs


def label(name):
    return NICE.get(name, name)


# --------------------------------------------------------------------------------------- tables
def build_markdown(runs):
    L = ["# Member 4 - results (auto-generated)", "",
         "Protocol: fitting, layer choice and thresholds use 5-fold out-of-fold predictions on train+val; the test split "
         "is scored once. Intervals are 95% bootstraps over solutions. Numbers are empirical estimates, not causal proof.", ""]
    for name, r in runs.items():
        m = r["res"]["meta"]
        L.append(f"- **{label(name)}**: layer {m['layer']} of {m['num_layers']} ({m['layer_selected_by']}), "
                 f"hidden {m['hidden_dim']}, test = {m['n_test']} solutions ({m['n_test_faulty']} faulty).")
    L += ["", "## 1. Threshold-free results (primary)", "",
          "| Model | Method | Step AUROC [95% CI] | AUROC without answer lines | Top-step accuracy % [95% CI] |", "|---|---|---|---|---|"]
    for name, r in runs.items():
        for meth in ("probe", "position_only", "length_only", "position+length"):
            x, ci = r["res"]["methods"][meth], r["res"]["ci95"][meth]
            a2 = x["step_auroc_no_answer_lines"]
            L.append(f"| {label(name)} | {METHOD_LABEL[meth]} | {dec_ci(x['step_auroc'], ci['step_auroc'])} | "
                     f"{'n/a' if a2 is None else f'{a2:.3f}'} | {pct_ci(x['top_step_acc'], ci['top_step_acc'])} |")
    L += ["", "Top-step accuracy = when the solution is known to be faulty, the highest-scoring step is the gold first-error step.", ""]
    ref = {n: r["res"]["reference"] for n, r in runs.items()}
    L += ["| Model | Random step (expected) | Modal step from train+val | Always answer 'correct' (joint) |", "|---|---|---|---|"]
    for n, x in ref.items():
        L.append(f"| {label(n)} | {pct(x['random_step_expected_top_step_acc_test'])} | "
                 f"{pct(x['modal_step_top_step_acc_test'])} (step {x['modal_first_error_step_from_train_val']}) | "
                 f"{pct(x['always_correct_joint_acc_test'])} |")
    for op, title in (("balanced", "2. Operating point 'balanced' (maximise mean of verdict and first-error on OOF)"),
                      ("joint", "3. Operating point 'joint' (maximise joint accuracy on OOF)")):
        L += ["", f"## {title}", "", "| Model | Method | Threshold | Verdict % | First-error % | Joint % |", "|---|---|---|---|---|---|"]
        for name, r in runs.items():
            for meth in ("probe", "position_only", "length_only", "position+length"):
                o = r["res"]["methods"][meth]["operating_points"][op]
                ci = r["res"]["ci95"][meth]
                L.append(f"| {label(name)} | {METHOD_LABEL[meth]} | {o['threshold']:.3f} | "
                         f"{pct_ci(o['test']['verdict'], ci[f'{op}/verdict'])} | {pct_ci(o['test']['first_error'], ci[f'{op}/first_error'])} | "
                         f"{pct_ci(o['test']['joint'], ci[f'{op}/joint'])} |")
    L += ["", "Joint accuracy cannot beat the 'always correct' reference on a 50/50 test set unless first-error accuracy is also high; "
          "this is why the balanced operating point is reported next to it.", "",
          "## 4. Paired difference, probe minus baseline (95% CI; interval excluding 0 = clear difference)", "",
          "| Model | Baseline | Step AUROC | Top-step acc (pts) | Balanced verdict (pts) | Balanced first-error (pts) |", "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for meth, d in r["res"]["paired_probe_minus_baseline_ci95"].items():
            f = lambda k, s=100: f"[{s * d[k][0]:+.1f}, {s * d[k][1]:+.1f}]"
            L.append(f"| {label(name)} | {METHOD_LABEL[meth]} | [{d['step_auroc'][0]:+.3f}, {d['step_auroc'][1]:+.3f}] | "
                     f"{f('top_step_acc')} | {f('balanced/verdict')} | {f('balanced/first_error')} |")
    L += ["", "## 5. Length audit of the probe (observational)", "",
          "| Model | rho(length, score), correct steps [95% CI] | AUROC short | AUROC mid | AUROC long |", "|---|---|---|---|---|"]
    for name, r in runs.items():
        a = r["res"]["length_audit_probe"]
        c = a["correct_steps"]
        t = a["auroc_within_length_tertile"]
        rho = "n/a" if not c else (f"{c['rho']:.2f} [{c['rho_ci95'][0]:.2f}, {c['rho_ci95'][1]:.2f}]" if "rho_ci95" in c else f"{c['rho']:.2f}")
        L.append(f"| {label(name)} | {rho} | {t['tertile1']} | {t['tertile2']} | {t['tertile3']} |")
    L += ["", "A positive rho on correct steps means longer steps get higher error scores even when they are correct. "
          "Falling AUROC from short to long steps means errors are harder to detect in long steps.", ""]
    if any(r["stress"] for r in runs.values()):
        L += ["## 6. Padding stress test (paired, same step with and without added text)", "",
              "| Model | Padding | Group | n | +tokens | Mean change in error score [95% CI] | flagged->unflagged | unflagged->flagged |",
              "|---|---|---|---|---|---|---|---|"]
        for name, r in runs.items():
            if not r["stress"]:
                continue
            for tpl, groups in r["stress"]["results"].items():
                for grp, c in groups.items():
                    L.append(f"| {label(name)} | {tpl} | {grp} | {c['n']} | {c['mean_added_tokens']:.0f} | "
                             f"{c['mean_delta']:+.3f} [{c['mean_delta_ci95'][0]:+.3f}, {c['mean_delta_ci95'][1]:+.3f}] | "
                             f"{c['balanced_flag_on_to_off']} | {c['balanced_flag_off_to_on']} |")
        L += ["", "Added text changes meaning as well as length: this shows sensitivity to the intervention, not proof of length bias.", ""]
    if any(r["audit"] for r in runs.values()):
        L += ["## 7. Data audit", ""]
        for name, r in runs.items():
            if r["audit"]:
                L.append(f"- {label(name)}: **{r['audit']['status']}**; leakage overlaps {r['audit']['leakage_overlaps']}; parity: "
                         f"{r['audit']['parity'] if isinstance(r['audit']['parity'], str) else 'worst cosine ' + str(r['audit']['parity']['worst_cosine'])}")
        L.append("")
    L += ["## Limitations", "",
          "- Only 50 faulty test solutions: intervals are about +/-13 points; do not rank differences smaller than that.",
          "- The 'balanced' operating point was declared after seeing that 'joint' is degenerate on 50/50 data; report both.",
          "- Length findings are observational; the padding test changes wording as well as length.",
          "- Layer choice used out-of-fold AUROC on train+val only; neighbouring layers are usually within noise.", ""]
    return "\n".join(L)


def write_csv(runs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "method", "step_auroc", "step_auroc_lo", "step_auroc_hi", "top_step_acc", "top_lo", "top_hi",
                    "operating_point", "threshold", "verdict", "first_error", "joint"])
        for name, r in runs.items():
            for meth, x in r["res"]["methods"].items():
                ci = r["res"]["ci95"][meth]
                for op, o in x["operating_points"].items():
                    w.writerow([name, r["res"]["meta"]["layer"], meth, x["step_auroc"], *ci["step_auroc"], x["top_step_acc"],
                                *ci["top_step_acc"], op, o["threshold"], o["test"]["verdict"], o["test"]["first_error"], o["test"]["joint"]])


def write_latex(runs, path):
    rows = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llccccc}", r"\toprule",
            r"Model & Method & AUROC & Top-step (\%) & Verdict (\%) & First-error (\%) & Joint (\%) \\", r"\midrule"]
    for name, r in runs.items():
        for meth in ("probe", "position_only", "length_only", "position+length"):
            x = r["res"]["methods"][meth]
            b = x["operating_points"]["balanced"]["test"]
            nm = label(name) if meth == "probe" else ""
            rows.append(f"{nm} & {METHOD_LABEL[meth]} & {x['step_auroc']:.3f} & {100 * x['top_step_acc']:.1f} & "
                        f"{100 * b['verdict']:.1f} & {100 * b['first_error']:.1f} & {100 * b['joint']:.1f} \\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows += [r"\end{tabular}",
             r"\caption{Test-set results (100 solutions, 50 faulty). Layer chosen by out-of-fold AUROC on train+val; "
             r"thresholds chosen out-of-fold (balanced operating point). Top-step: highest-scoring step equals the gold "
             r"first-error step given a faulty solution.}", r"\label{tab:member4_main}", r"\end{table}"]
    Path(path).write_text("\n".join(rows), encoding="utf-8")


# --------------------------------------------------------------------------------------- figures
def fig_sweep(runs, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    drawn = 0
    for i, (name, r) in enumerate(runs.items()):
        if not r["sweep"]:
            continue
        rows = r["sweep"]["rows"]
        n = r["sweep"]["num_layers"]
        x = [row["layer"] / n for row in rows]
        yv = [row["oof_step_auroc"] for row in rows]
        c = MODEL_COLORS[i % 3]
        ax.plot(x, yv, color=c, lw=2, marker=MODEL_MARKERS[i % 3], ms=6, label=label(name))
        sel = r["sweep"]["selected_layer"]
        sy = next(row["oof_step_auroc"] for row in rows if row["layer"] == sel)
        ax.scatter([sel / n], [sy], s=140, facecolor="none", edgecolor=c, lw=2, zorder=5)
        ax.annotate(f"L{sel}", (sel / n, sy), textcoords="offset points", xytext=(0, 12), ha="center", color=INK2, fontsize=9)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Relative depth (layer / total layers)")
    ax.set_ylabel("Out-of-fold step AUROC (train+val)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.06 * (hi - lo))  # headroom for the selected-layer labels
    ax.set_title("Where the error signal lives: layer sweep, no test data used", loc="left", fontsize=11, color=INK, pad=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_main(runs, path):
    from matplotlib.lines import Line2D
    names = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    panels = [("step_auroc", "Step AUROC (test)", (0.4, 1.1)), ("top_step_acc", "Top-step accuracy, known-faulty solutions (test)", (0, 1.1))]
    methods = ("probe", "position_only", "length_only", "position+length")
    bw = 0.19
    for ax, (key, title, ylim) in zip(axes, panels):
        for gi, name in enumerate(names):
            for mi, meth in enumerate(methods):
                v = runs[name]["res"]["methods"][meth][key]
                lo, hi = runs[name]["res"]["ci95"][meth][key]
                x = gi + (mi - 1.5) * (bw + 0.02)
                ax.bar(x, v, width=bw, color=METHOD_COLOR[meth], edgecolor=SURFACE, linewidth=1.5, label=METHOD_LABEL[meth] if gi == 0 else None)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK2, lw=1, capsize=2.5)
                ax.text(x, hi + 0.015 * (ylim[1] - ylim[0]), f"{v:.2f}" if key == "step_auroc" else f"{100 * v:.0f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
        if key == "top_step_acc":
            for gi, name in enumerate(names):
                rnd = runs[name]["res"]["reference"]["random_step_expected_top_step_acc_test"]
                ax.hlines(rnd, gi - 0.45, gi + 0.45, colors=INK2, linestyles="--", lw=1)
        else:
            ax.axhline(0.5, color=INK2, ls=":", lw=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([label(n) for n in names])
        ax.set_ylim(*ylim)
        ax.set_yticks([t for t in ax.get_yticks() if ylim[0] <= t <= 1.0])
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOR[m]) for m in methods]
    handles += [Line2D([0], [0], color=INK2, ls=":", lw=1), Line2D([0], [0], color=INK2, ls="--", lw=1)]
    names_ = [METHOD_LABEL[m] for m in methods] + ["Chance AUROC (0.5)", "Random step (expected)"]
    fig.legend(handles, names_, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_length(runs, path):
    names = [n for n in runs if runs[n]["res"].get("length_audit_probe")]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    bw = 0.34
    for i, name in enumerate(names):
        t = runs[name]["res"]["length_audit_probe"]["auroc_within_length_tertile"]
        vals = [t["tertile1"], t["tertile2"], t["tertile3"]]
        for j, v in enumerate(vals):
            if v is None:
                continue
            x = j + (i - (len(names) - 1) / 2) * (bw + 0.03)
            ax.bar(x, v, width=bw, color=MODEL_COLORS[i % 3], edgecolor=SURFACE, linewidth=1.5, label=label(name) if j == 0 else None)
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0.5, color=INK2, ls=":", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["short steps", "medium steps", "long steps"])
    ax.set_ylim(0.4, 1.1)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Step AUROC inside length tertile (test)")
    ax.set_title("Is the signal just step length? AUROC within matched-length thirds", loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=max(1, len(names)))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="name=folder pairs, e.g. qwen25=m4_outputs/qwen25")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    runs = load_runs(a.runs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(build_markdown(runs), encoding="utf-8")
    write_csv(runs, out / "main_results.csv")
    write_latex(runs, out / "table_main.tex")
    fig_sweep(runs, out / "fig_layer_sweep.png")
    fig_main(runs, out / "fig_main.png")
    fig_length(runs, out / "fig_length.png")
    print(f"wrote report.md, main_results.csv, table_main.tex and figures to {out}")


if __name__ == "__main__":
    main()
