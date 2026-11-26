"""Reproduce the CPU benchmark, one fresh process per backend and split."""
import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TABPFN_NO_BROWSER"] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "benchmarks/gss_cat.csv.gz"
METHODS = ["independent", "bootstrap", "cart", "ctgan", "tabpfn-v2", "tabpfn-v3.5-fast", "tabpfn-v3.5"]
NUMERIC = ["age", "tvhours"]
CATEGORIES = ["year", "marital", "race", "rincome", "partyid", "relig", "denom"]


def load_split(seed):
    # "None" is a substantive religion/denomination response.
    data = pd.read_csv(DATA, keep_default_na=False, na_values=[""]).dropna()
    data[CATEGORIES] = data[CATEGORIES].astype(str)
    data[NUMERIC] = data[NUMERIC].astype(int)
    rows = data.sample(n=1536, random_state=seed)
    return rows.iloc[:512].reset_index(drop=True), rows.iloc[512:].reset_index(drop=True)


def cart_sample(train, seed):
    """Fit one tree per conditional, then draw a training response from each leaf."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    rng = np.random.default_rng(seed)
    generated = pd.DataFrame(index=train.index)
    for i, col in enumerate(train):
        source = train[col].to_numpy()
        if i == 0:
            generated[col] = rng.choice(source, len(train))
            continue
        x = pd.get_dummies(train.iloc[:, :i], dtype=float)
        query = pd.get_dummies(generated, dtype=float).reindex(columns=x.columns, fill_value=0)
        tree_type = DecisionTreeClassifier if col in CATEGORIES else DecisionTreeRegressor
        tree = tree_type(min_samples_leaf=12, random_state=seed).fit(x, source)
        leaves, targets = tree.apply(x), tree.apply(query)
        output = np.empty(len(train), dtype=source.dtype)
        for leaf in np.unique(targets):
            mask = targets == leaf
            output[mask] = rng.choice(source[leaves == leaf], mask.sum())
        generated[col] = output
    return generated


def synthesize(train, method, seed):
    n = len(train)
    if method == "cart":
        return cart_sample(train, seed)
    if method == "bootstrap":
        return train.sample(n, replace=True, random_state=seed)
    if method == "independent":
        rng = np.random.default_rng(seed)
        return pd.DataFrame({col: rng.choice(train[col], n) for col in train})
    if method == "ctgan":
        from generator import DataGenerator
        generator = DataGenerator()
        options = dict(epochs=500, batch_size=100, pac=10, seed=seed)
    else:
        from tabpfn_generator import TabPFNGenerator
        generator = TabPFNGenerator(version=method.removeprefix("tabpfn-"), seed=seed)
        options = {}
    data = generator.prepare_data(train, CATEGORIES)
    if not generator.train(data, **options):
        raise RuntimeError("training failed")
    return generator.generate(n)


def tv(a, b):
    p, q = a.value_counts(normalize=True).align(b.value_counts(normalize=True), fill_value=0)
    return float((p - q).abs().sum() / 2)


def metrics(train, held, syn, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if syn is None or syn.shape != train.shape or list(syn) != list(train):
        raise ValueError("generated table shape or schema mismatch")
    if syn.isna().any().any():
        raise ValueError("unexpected null on this complete-case benchmark")
    for col in CATEGORIES:
        if not syn[col].isin(train[col]).all():
            raise ValueError(f"unknown category in {col}")
    for col in NUMERIC:
        if not syn[col].between(train[col].min(), train[col].max()).all():
            raise ValueError(f"numeric range violation in {col}")

    real, fake = held.copy(), syn.copy()
    for col in NUMERIC:
        cutoffs = np.unique(train[col].quantile([.2, .4, .6, .8]))
        real[col] = np.searchsorted(cutoffs, real[col])
        fake[col] = np.searchsorted(cutoffs, fake[col])
    marginal = [tv(real[col], fake[col]) for col in train]
    pairwise = [tv(real[list(pair)], fake[list(pair)]) for pair in combinations(train, 2)]

    def marital_auc(frame):
        inputs = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), [c for c in CATEGORIES if c != "marital"]),
            ("numeric", StandardScaler(), NUMERIC),
        ])
        model = make_pipeline(inputs, LogisticRegression(max_iter=500, random_state=seed))
        model.fit(frame, frame.marital == "Married")
        return float(roc_auc_score(held.marital == "Married", model.predict_proba(held)[:, 1]))

    overlap = pd.MultiIndex.from_frame(syn).isin(pd.MultiIndex.from_frame(train))
    return dict(marginal_tv=np.mean(marginal), pair_tv=np.mean(pairwise),
                religion_denom_tv=tv(real[["relig", "denom"]], fake[["relig", "denom"]]),
                tstr_auc=marital_auc(syn), real_train_auc=marital_auc(train),
                train_match=float(overlap.mean()))


def worker(method, seed, path):
    train, held = load_split(seed)
    record = dict(method=method, seed=seed)
    started = time.perf_counter()
    try:
        syn = synthesize(train, method, seed)
        elapsed = time.perf_counter() - started
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        record.update(metrics(train, held, syn, seed))
        record.update(status="ok", seconds=elapsed, peak_mib=peak_mib)
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc))
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record["status"] == "ok"


def main():
    import tempfile
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.methods[0], args.seeds[0], args.output)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            for method in args.methods:
                path = Path(tmp) / f"{method}-{seed}.json"
                run = subprocess.run([sys.executable, __file__, "--worker", "--methods", method,
                                      "--seeds", str(seed), "--output", str(path)])
                record = json.loads(path.read_text()) if path.exists() else dict(
                    method=method, seed=seed, status="error", error=f"process exit {run.returncode}")
                results.append(record)
                args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
                print(method, seed, record["status"], flush=True)
    return all(r["status"] == "ok" for r in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
