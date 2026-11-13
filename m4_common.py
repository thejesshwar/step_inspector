"""m4_common.py - shared utilities for Member 4 (diagnostic benchmarks & meta-evaluation).

Everything here works on the cached tensors produced by extract_and_cache.py and never touches
the test split for training, layer choice or threshold choice.

Protocol (declared before looking at any test number):
  * Pool = train + validation (550 solutions). All model fitting, layer selection and
    threshold selection use 5-fold out-of-fold (OOF) predictions on the pool.
  * Test (100 solutions) is scored once, with the fold-ensemble of probes.
  * Primary metrics are threshold-free (step AUROC, top-step accuracy on known-faulty solutions).
    Threshold-based metrics are reported at two operating points chosen on OOF data:
    'joint' (maximise joint accuracy) and 'balanced' (maximise mean of verdict and first-error).
  * Uncertainty = bootstrap over solutions (steps of one solution are never treated as independent).
"""
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REQUIRED = ("step_embeddings_mean", "step_targets", "step_masks", "gold_first_errors", "step_token_lengths")
# Same regex as the team's spatial_pooling.py (verified 650/650 on MR-GSM8K).
STEP_PATTERN = re.compile(r"(Step\s+\d+[:\.\-].*?)(?=(?:\n\s*Step\s+\d+[:\.\-]|$))", re.DOTALL)
ANSWER_LINE = re.compile(r"####|The answer is", re.IGNORECASE)
ARITHMETIC = re.compile(r"\d\s*[+*/=\-x×]\s*\d|\d\s*=\s*")
METHODS = ("probe", "position_only", "length_only", "position+length")
OBJECTIVES = ("joint", "balanced")

# (num_layers, hidden_dim) -> (tag, Hugging Face id)
KNOWN_MODELS = {(24, 896): ("qwen25", "Qwen/Qwen2.5-0.5B-Instruct"),
                (28, 1024): ("qwen3", "Qwen/Qwen3-0.6B")}


# ----------------------------------------------------------------------------------- data
def load_cache(path):
    try:  # mmap keeps RAM low for the 28-layer x 1024-d cache
        c = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        c = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED if k not in c]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    n = len(c["gold_first_errors"])
    for k in REQUIRED:
        if len(c[k]) != n:
            raise ValueError(f"{path}: inconsistent batch size for {k}")
    return {k: c[k] for k in REQUIRED}


def load_splits(cache_dir):
    d = Path(cache_dir)
    return [load_cache(d / f"{s}_cached.pt") for s in ("train", "val", "test")]


def identify_model(cache, tag=None):
    _, n_layers, _, dim = cache["step_embeddings_mean"].shape
    default_tag, model_id = KNOWN_MODELS.get((n_layers, dim), (f"model_{n_layers}x{dim}", None))
    return (tag or default_tag), model_id, n_layers, dim


def layer_arrays(cache, layer):
    """1-based layer -> numpy arrays (emb [N,M,D], targets, masks, gold, lens)."""
    return (cache["step_embeddings_mean"][:, layer - 1].float().numpy(),
            cache["step_targets"].numpy().astype(np.float32),
            cache["step_masks"].numpy().astype(np.float32),
            cache["gold_first_errors"].numpy().astype(int),
            cache["step_token_lengths"].numpy().astype(int))


def pool_layer(train, val, layer):
    a, b = layer_arrays(train, layer), layer_arrays(val, layer)
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def parse_steps(candidate):
    return [(m.start(), m.end()) for m in STEP_PATTERN.finditer(candidate)]


def load_dataset_split(path, split="test"):
    import json
    root = json.loads(Path(path).read_text(encoding="utf-8"))
    return root["splits"][split]


def check_alignment(examples, lens):
    """Dataset rows must line up with the cache rows (same order, same number of steps)."""
    if len(examples) != len(lens):
        raise ValueError(f"dataset has {len(examples)} rows but cache has {len(lens)}")
    for i, ex in enumerate(examples):
        if len(parse_steps(ex["candidate"])) != int((lens[i] > 0).sum()):
            raise ValueError(f"row {i}: step count in dataset != step count in cache")


def answer_line_mask(examples, shape):
    out = np.zeros(shape, dtype=bool)
    for i, ex in enumerate(examples):
        for j, m in enumerate(STEP_PATTERN.finditer(ex["candidate"])):
            if j < shape[1] and ANSWER_LINE.search(m.group()):
                out[i, j] = True
    return out


def trusted_mask(lens, gold):
    """Valid steps up to and including the first error. Steps after the error are untrusted."""
    pos = np.arange(1, lens.shape[1] + 1)[None, :]
    return (lens > 0) & ((gold[:, None] == 0) | (pos <= gold[:, None]))


# ----------------------------------------------------------------------------------- metrics
def decode(prob, lens, thr):
    """0 = solution judged correct, otherwise the earliest step whose error probability > thr."""
    flagged = (prob > thr) & (lens > 0)
    return np.where(flagged.any(1), flagged.argmax(1) + 1, 0)


def metrics(pred, gold):
    faulty = gold > 0
    return {"verdict": float(np.mean((pred > 0) == faulty)),
            "first_error": float(np.mean(pred[faulty] == gold[faulty])),
            "joint": float(np.mean(pred == gold))}


def objective(m, obj):
    return m["joint"] if obj == "joint" else (m["verdict"] + m["first_error"]) / 2


def pick_threshold(prob, gold, lens, obj):
    grid = np.linspace(0.05, 0.95, 37)
    scores = [objective(metrics(decode(prob, lens, t), gold), obj) for t in grid]
    best = max(range(len(grid)), key=lambda j: (scores[j], -abs(grid[j] - 0.5)))
    return float(grid[best])


def top_step_hits(prob, lens, gold):
    """1 if the highest-scoring valid step is the gold first-error step (meaningful for faulty rows)."""
    p = np.where(lens > 0, prob, -np.inf)
    return (p.argmax(1) + 1 == gold).astype(float)


def safe_auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


# ----------------------------------------------------------------------------------- models
class StepProbe(nn.Module):
    """Same architecture as the team's probe (run_role3_depth.StepProbe): D -> D/4 -> LN -> GELU -> 1.
    Attribute is called `head` so a team checkpoint's state_dict keys line up if ever needed."""

    def __init__(self, hidden_dim):
        super().__init__()
        h = hidden_dim // 4
        self.head = nn.Sequential(nn.Linear(hidden_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.head(x).squeeze(-1)


def make_mlp_fitter(epochs=20, lr=1e-3, weight_decay=1e-2, batch=32):
    def fit(X, y, m, seed):
        torch.manual_seed(seed)
        X, y, m = (torch.as_tensor(a, dtype=torch.float32) for a in (X, y, m))
        net = StepProbe(X.shape[-1])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            net.train()
            for b in torch.randperm(len(X), generator=gen).split(batch):
                loss = (F.binary_cross_entropy_with_logits(net(X[b]), y[b], reduction="none") * m[b]).sum()
                loss = loss / m[b].sum().clamp_min(1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()

        def predict(Z):
            with torch.no_grad():
                return torch.sigmoid(net(torch.as_tensor(Z, dtype=torch.float32))).numpy()
        return predict
    return fit


def lr_fitter(X, y, m, seed):
    keep = m.reshape(-1) > 0
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X.reshape(-1, X.shape[-1])[keep], y.reshape(-1)[keep].astype(int))
    return lambda Z: clf.predict_proba(Z.reshape(-1, Z.shape[-1]))[:, 1].reshape(Z.shape[:2])


def baseline_features(lens):
    """Features that carry NO reasoning content: absolute step position and/or log token length."""
    n, M = lens.shape
    pos = np.broadcast_to(np.eye(M, dtype=np.float32), (n, M, M)).copy()
    loglen = np.log1p(lens).astype(np.float32)[..., None]
    return {"position_only": pos, "length_only": loglen, "position+length": np.concatenate([pos, loglen], -1)}


def cv_fit(X, y, m, gold, fitter, folds=5, seeds=(0,)):
    """Out-of-fold probabilities on the pool + the list of fitted predictors (for test-time ensembling)."""
    oof = np.zeros(X.shape[:2])
    predictors = []
    for s in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=s)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(X)), gold > 0)):
            predict = fitter(X[tr], y[tr], m[tr], s * 100 + k)
            oof[te] += predict(X[te]) / len(seeds)
            predictors.append(predict)
    return oof, predictors


def ensemble_predict(predictors, Z):
    return np.mean([p(Z) for p in predictors], axis=0)


# ----------------------------------------------------------------------------------- bootstrap
def method_summary(prob_test, oof_pool, ops, gold_t, lens_t, y_t, trusted_t):
    """Per-solution outcomes needed for fast clustered bootstrapping."""
    sid, step = np.nonzero(trusted_t)
    out = {"ops": {}, "hit": top_step_hits(prob_test, lens_t, gold_t),
           "y": y_t[sid, step], "p": prob_test[sid, step], "sid": sid}
    for obj, thr in ops.items():
        pred = decode(prob_test, lens_t, thr)
        out["ops"][obj] = {"verdict_ok": ((pred > 0) == (gold_t > 0)).astype(float),
                           "exact": (pred == gold_t).astype(float), "pred": pred}
    return out


def _stats(m, c, n, cf, faulty):
    s = {}
    for obj, d in m["ops"].items():
        s[f"{obj}/verdict"] = float((c * d["verdict_ok"]).sum() / n)
        s[f"{obj}/first_error"] = float((c * d["exact"] * faulty).sum() / cf)
        s[f"{obj}/joint"] = float((c * d["exact"]).sum() / n)
    s["top_step_acc"] = float((c * m["hit"] * faulty).sum() / cf)
    try:
        s["step_auroc"] = float(roc_auc_score(m["y"], m["p"], sample_weight=c[m["sid"]]))
    except ValueError:
        s["step_auroc"] = float("nan")
    return s


def bootstrap_all(methods, gold, B=2000, seed=0, reference="probe"):
    """Resample SOLUTIONS (not steps). Same resamples for every method -> valid paired differences."""
    n = len(gold)
    faulty = (gold > 0).astype(float)
    rng = np.random.RandomState(seed)
    acc = {name: {} for name in methods}
    for _ in range(B):
        c = np.bincount(rng.randint(0, n, n), minlength=n).astype(float)
        cf = float((c * faulty).sum())
        if cf == 0:
            continue
        for name, m in methods.items():
            for k, v in _stats(m, c, n, cf, faulty).items():
                acc[name].setdefault(k, []).append(v)
    arr = {name: {k: np.array(v) for k, v in d.items()} for name, d in acc.items()}
    pct = lambda v: [round(float(x), 4) for x in np.nanpercentile(v, [2.5, 97.5])]
    ci = {name: {k: pct(v) for k, v in d.items()} for name, d in arr.items()}
    paired = {name: {k: pct(arr[reference][k] - arr[name][k]) for k in arr[reference]}
              for name in arr if name != reference}
    point = {name: _stats(m, np.ones(n), n, float(faulty.sum()), faulty) for name, m in methods.items()}
    return point, ci, paired


# ----------------------------------------------------------------------------------- length audit
def length_audit(lens, gold, targets, probs, answer_lines, B=500, seed=0):
    """Observational audit on trusted steps, answer lines excluded (they are short and always correct)."""
    n = len(gold)
    tm = trusted_mask(lens, gold) & ~answer_lines
    pos = np.arange(1, lens.shape[1] + 1)[None, :]
    err = tm & (gold[:, None] > 0) & (pos == gold[:, None])
    cor = tm & ~err
    out = {"n_correct_steps": int(cor.sum()), "n_first_error_steps": int(err.sum()),
           "note": "answer lines excluded; rho CI clustered by solution; observational, not causal"}
    for name, mask in (("correct_steps", cor), ("first_error_steps", err)):
        if mask.sum() < 3:
            out[name] = None
            continue
        out[name] = {"rho": round(float(spearmanr(lens[mask], probs[mask]).statistic), 4)}
    if cor.sum() >= 3:
        per = [(lens[i][cor[i]], probs[i][cor[i]]) for i in range(n)]
        rng = np.random.RandomState(seed)
        boots = []
        for _ in range(B):
            idx = rng.randint(0, n, n)
            L = np.concatenate([per[i][0] for i in idx])
            P = np.concatenate([per[i][1] for i in idx])
            if len(np.unique(L)) > 1 and len(np.unique(P)) > 1:
                boots.append(spearmanr(L, P).statistic)
        if boots:
            out["correct_steps"]["rho_ci95"] = [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])]
    edges = np.quantile(lens[tm], [1 / 3, 2 / 3])
    bins = np.digitize(lens, edges)
    out["tertile_token_cutoffs"] = [float(e) for e in edges]
    out["auroc_within_length_tertile"] = {
        f"tertile{b + 1}": (None if safe_auroc(targets[tm & (bins == b)], probs[tm & (bins == b)]) is None
                            else round(safe_auroc(targets[tm & (bins == b)], probs[tm & (bins == b)]), 4))
        for b in range(3)}
    return out
