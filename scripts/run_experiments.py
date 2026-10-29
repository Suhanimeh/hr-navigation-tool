#!/usr/bin/env python
"""Reproducible model comparison for all three HR tasks.

Usage:
    python scripts/run_experiments.py                       # all tasks, data/ -> results/
    python scripts/run_experiments.py --task attrition
    python scripts/run_experiments.py --data-dir path/to/csvs --seed 7

Protocol (identical for every task)
    1. Stratified 80/20 train/test split (fixed seed).
    2. 5-fold stratified CV on the TRAIN split only -> model comparison table.
    3. Best model chosen by CV PR-AUC (never by test score, never by accuracy).
    4. All models refit on the full train split and scored ONCE on the test split.
    5. Diagnostics for the selected model: confusion matrix, ROC/PR curves, threshold
       table, permutation importance, most-confident errors, group fairness diagnostic,
       and an ablation that adds the sensitive columns back in.
Everything is written to results/<task>/ together with run metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hr_analytics import data as D  # noqa: E402
from hr_analytics import evaluation as E  # noqa: E402
from hr_analytics import modeling as M  # noqa: E402
from hr_analytics.config import SEED, TASKS  # noqa: E402


def run_task(name: str, data_dir: Path, out_root: Path, seed: int) -> dict:
    task = TASKS[name]
    path = data_dir / task.default_filename
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    df = D.load_csv(path)
    X, y = D.prepare_xy(df, task, use_sensitive=False)
    X_tr, X_te, y_tr, y_te = D.split(X, y, seed=seed)
    print(f"\n=== {task.title}: {len(df)} rows, {X.shape[1]} features, "
          f"positive rate {y.mean():.3f}; train {len(X_tr)} / test {len(X_te)}")

    pipes = M.make_pipelines(X_tr, seed=seed)
    cv = E.cross_validate_models(pipes, X_tr, y_tr, seed=seed)
    best = E.select_best(cv, task.selection_metric)
    fitted, test = E.fit_and_score_test(pipes, X_tr, y_tr, X_te, y_te)
    cv.to_csv(out / "cv_results.csv")
    test.to_csv(out / "test_results.csv")
    print(cv[["pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]].round(3))
    print(f"Selected by CV {task.selection_metric}: {best}")

    proba = fitted[best].predict_proba(X_te)[:, 1]

    # confusion matrix, ROC, PR
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(E.confusion(y_te, proba), display_labels=["No", "Yes"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"{best} - test set (threshold 0.5)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, p in fitted.items():
        pr = p.predict_proba(X_te)[:, 1]
        RocCurveDisplay.from_predictions(y_te, pr, name=n, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_te, pr, name=n, ax=axes[1])
    axes[0].set_title("ROC (test)"); axes[1].set_title("Precision-Recall (test)")
    fig.tight_layout(); fig.savefig(out / "roc_pr_curves.png", dpi=150); plt.close(fig)

    # threshold trade-off, errors, importance
    E.threshold_table(y_te, proba).to_csv(out / "threshold_table.csv", index=False)
    fp, fn = E.error_analysis(X_te, y_te, proba)
    fp.to_csv(out / "errors_false_positives.csv"); fn.to_csv(out / "errors_false_negatives.csv")
    imp = E.importance_table(fitted[best], X_te, y_te, seed=seed)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#2E5E8C")
    ax.set_xlabel("Drop in PR-AUC when shuffled"); ax.set_title(f"Permutation importance - {best}")
    fig.tight_layout(); fig.savefig(out / "permutation_importance.png", dpi=150); plt.close(fig)

    # fairness diagnostic on the sensitive group column (even though it is NOT a model input)
    fair = None
    if task.group_column and task.group_column in df.columns:
        fair = E.group_diagnostic(df.loc[X_te.index, task.group_column], y_te, proba)
        fair.to_csv(out / "group_diagnostic.csv")

    # ablation: do sensitive features change performance?
    Xs, _ = D.prepare_xy(df, task, use_sensitive=True)
    Xs_tr, Xs_te = Xs.loc[X_tr.index], Xs.loc[X_te.index]
    est = M.get_estimators(seed)[best]
    sens_pipe = M.make_pipeline(best, est, Xs_tr).fit(Xs_tr, y_tr)
    sens = E.score_all(y_te, sens_pipe.predict_proba(Xs_te)[:, 1])
    ablation = {"without_sensitive_pr_auc": test.loc[best, "pr_auc"],
                "with_sensitive_pr_auc": sens["pr_auc"],
                "sensitive_columns": [c for c in task.sensitive_columns if c in df.columns]}
    print("Ablation (PR-AUC test):", {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in ablation.items()})

    meta = {
        "task": name, "rows": int(len(df)), "n_features": int(X.shape[1]),
        "features": list(X.columns), "positive_rate": float(y.mean()),
        "seed": seed, "selected_model": best, "selection_metric": task.selection_metric,
        "ablation": ablation,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"meta": meta, "cv": cv, "test": test, "fair": fair, "imp": imp}


def fmt(df: pd.DataFrame, cols) -> str:
    return df[cols].round(3).to_markdown()


def write_summary(results: dict, out_root: Path) -> None:
    lines = ["# Experiment summary (auto-generated by scripts/run_experiments.py)\n"]
    for name, r in results.items():
        m = r["meta"]
        lines += [
            f"## {TASKS[name].title}",
            f"{m['rows']} rows, {m['n_features']} features, positive rate {m['positive_rate']:.1%}, "
            f"seed {m['seed']}, scikit-learn {m['sklearn']}.\n",
            f"**Selected model (CV {m['selection_metric']}): {m['selected_model']}**\n",
            "### 5-fold CV on train split (mean)",
            fmt(r["cv"], ["pr_auc", "roc_auc", "recall", "precision", "f1", "macro_f1", "accuracy"]),
            "\n### Held-out test split (threshold 0.5)",
            fmt(r["test"], ["pr_auc", "roc_auc", "recall", "precision", "f1", "macro_f1", "accuracy",
                            "fit_time_s"]),
            "\n### Top permutation importances (selected model)",
            r["imp"].head(8).round(4).to_markdown(index=False),
        ]
        if r["fair"] is not None:
            lines += ["\n### Group diagnostic (test, selected model)", r["fair"].round(3).to_markdown()]
        a = m["ablation"]
        lines += [f"\n### Ablation: adding sensitive columns {a['sensitive_columns']}",
                  f"Test PR-AUC {a['without_sensitive_pr_auc']:.3f} (excluded) vs "
                  f"{a['with_sensitive_pr_auc']:.3f} (included).\n"]
    (out_root / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=[*TASKS, "all"], default="all")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    names = list(TASKS) if a.task == "all" else [a.task]
    results = {}
    for n in names:
        try:
            results[n] = run_task(n, a.data_dir, a.out_dir, a.seed)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {n}: {exc}", file=sys.stderr)
    if results:
        write_summary(results, a.out_dir)
        print(f"\nWrote {a.out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
