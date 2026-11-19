from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
RANDOM_SEED = 10000

# Years are treated as categorical identifiers, as described in the manuscript.
CATEGORICAL_COLS = [
    "播种年",
    "播种日期",
    "品种",
    "收获年份",
    "灌溉方式",
    "灌溉期",
    "耕作模式",
    "覆盖模式",
]
GROUP_COLS = ["播种年", "纬度", "经度"]

TARGET_FILES = {
    "yield": {
        "train": "Y_train_YD.xlsx",
        "test": "Y_test_YD.xlsx",
        "label": "Yield",
        "unit": "kg/ha",
    },
    "wue": {
        "train": "Y_train_WUE.xlsx",
        "test": "Y_test_WUE.xlsx",
        "label": "WUE",
        "unit": "kg/m3",
    },
}


@dataclass
class FoldFittedModels:
    rf: Pipeline
    xgb: Pipeline
    ridge: Ridge
    inner_splits: list[tuple[np.ndarray, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible RF-XGBoost-Ridge stacking analysis with leakage-aware validation."
    )
    parser.add_argument(
        "--target",
        choices=["yield", "wue", "all"],
        default="all",
        help="Outcome to model. 'all' runs yield and WUE sequentially.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["holdout", "record_cv", "grouped_cv", "all"],
        default=["holdout", "grouped_cv"],
        help=(
            "Validation procedures. holdout uses the supplied 80/20 files; "
            "record_cv is the record-level five-fold benchmark; grouped_cv "
            "holds complete site-years together."
        ),
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use one compact grid per learner for an installation/smoke test.",
    )
    return parser.parse_args()


def load_target(target_name: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    spec = TARGET_FILES[target_name]
    x_train = pd.read_excel(BASE_DIR / "X_train.xlsx").reset_index(drop=True)
    x_test = pd.read_excel(BASE_DIR / "X_test.xlsx").reset_index(drop=True)
    y_train = (
        pd.read_excel(BASE_DIR / spec["train"]).iloc[:, 0].astype(float).reset_index(drop=True)
    )
    y_test = (
        pd.read_excel(BASE_DIR / spec["test"]).iloc[:, 0].astype(float).reset_index(drop=True)
    )
    if len(x_train) != len(y_train):
        raise ValueError(
            f"Training X/y length mismatch for {target_name}: {len(x_train)} vs {len(y_train)}"
        )
    if len(x_test) != len(y_test):
        raise ValueError(
            f"Test X/y length mismatch for {target_name}: {len(x_test)} vs {len(y_test)}"
        )
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Training and test feature columns are not identical or not in the same order")
    return x_train, y_train, x_test, y_test


def make_group_key(x: pd.DataFrame) -> pd.Series:
    missing = [col for col in GROUP_COLS if col not in x.columns]
    if missing:
        raise ValueError(f"Missing site-year grouping columns: {missing}")
    return x[GROUP_COLS].astype(str).agg("|".join, axis=1)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [col for col in CATEGORICAL_COLS if col in x.columns]
    numeric_cols = [col for col in x.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(x: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(x)),
            ("model", estimator),
        ]
    )


def make_splits(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    validation: str,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if validation == "grouped":
        return list(GroupKFold(n_splits=n_splits).split(x, y, groups=groups))
    if validation == "record":
        return list(
            KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED).split(x, y)
        )
    raise ValueError(f"Unknown validation mode: {validation}")


def parameter_grids(fast: bool) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
    if fast:
        return (
            {
                "model__n_estimators": [60],
                "model__max_depth": [20],
                "model__min_samples_split": [2],
                "model__min_samples_leaf": [1],
            },
            {
                "model__n_estimators": [80],
                "model__max_depth": [5],
                "model__learning_rate": [0.1],
                "model__subsample": [0.8],
            },
            {"alpha": [1.0]},
        )
    return (
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 0.9],
        },
        {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    )


def tune_model(model, grid, x, y, splits, n_jobs: int) -> GridSearchCV:
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        cv=splits,
        scoring="r2",
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(x, y)
    return search


def fit_stacking(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    validation: str,
    cv_folds: int,
    n_jobs: int,
    fast: bool,
) -> FoldFittedModels:
    rf_grid, xgb_grid, ridge_grid = parameter_grids(fast)
    inner_splits = make_splits(x_train, y_train, groups_train, validation, cv_folds)

    rf_pipeline = build_model(
        x_train,
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=1),
    )
    xgb_pipeline = build_model(
        x_train,
        XGBRegressor(
            random_state=100,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
        ),
    )

    rf_search = tune_model(rf_pipeline, rf_grid, x_train, y_train, inner_splits, n_jobs)
    xgb_search = tune_model(xgb_pipeline, xgb_grid, x_train, y_train, inner_splits, n_jobs)

    rf_oof = cross_val_predict(
        rf_search.best_estimator_,
        x_train,
        y_train,
        cv=inner_splits,
        n_jobs=1,
    )
    xgb_oof = cross_val_predict(
        xgb_search.best_estimator_,
        x_train,
        y_train,
        cv=inner_splits,
        n_jobs=1,
    )
    stack_x = pd.DataFrame(
        np.column_stack([rf_oof, xgb_oof]),
        columns=["rf_prediction", "xgb_prediction"],
    )

    ridge_search = tune_model(
        Ridge(),
        ridge_grid,
        stack_x,
        y_train.reset_index(drop=True),
        inner_splits,
        n_jobs,
    )

    return FoldFittedModels(
        rf=rf_search.best_estimator_,
        xgb=xgb_search.best_estimator_,
        ridge=ridge_search.best_estimator_,
        inner_splits=inner_splits,
    )


def predict_models(models: FoldFittedModels, x: pd.DataFrame) -> dict[str, np.ndarray]:
    rf_pred = models.rf.predict(x)
    xgb_pred = models.xgb.predict(x)
    stack_pred = models.ridge.predict(np.column_stack([rf_pred, xgb_pred]))
    return {"RF": rf_pred, "XGBoost": xgb_pred, "Stacking": stack_pred}


def regression_metrics(y_true, prediction) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    prediction_arr = np.asarray(list(prediction), dtype=float)
    return {
        "n": int(len(y_true_arr)),
        "r2": float(r2_score(y_true_arr, prediction_arr)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_arr, prediction_arr))),
        "mae": float(mean_absolute_error(y_true_arr, prediction_arr)),
    }


def evaluate_holdout(
    x_train,
    y_train,
    x_test,
    y_test,
    cv_folds,
    n_jobs,
    fast,
):
    models = fit_stacking(
        x_train,
        y_train,
        make_group_key(x_train),
        validation="record",
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        fast=fast,
    )
    predictions = predict_models(models, x_test)
    metrics = {name: regression_metrics(y_test, pred) for name, pred in predictions.items()}
    table = x_test.copy()
    table["actual"] = y_test.to_numpy()
    table["site_year_group"] = make_group_key(x_test).to_numpy()
    for name, pred in predictions.items():
        table[f"pred_{name}"] = pred
    return metrics, table, models


def evaluate_cv(
    x: pd.DataFrame,
    y: pd.Series,
    validation: str,
    cv_folds: int,
    n_jobs: int,
    fast: bool,
    label: str,
):
    groups = make_group_key(x)
    outer_splits = make_splits(x, y, groups, validation, cv_folds)
    predictions = {
        "RF": np.full(len(x), np.nan),
        "XGBoost": np.full(len(x), np.nan),
        "Stacking": np.full(len(x), np.nan),
    }
    fold_ids = np.full(len(x), -1, dtype=int)

    for fold_id, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        print(f"[{label}] outer fold {fold_id}/{len(outer_splits)}", flush=True)
        models = fit_stacking(
            x.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            groups.iloc[train_idx].reset_index(drop=True),
            validation=validation,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            fast=fast,
        )
        fold_predictions = predict_models(models, x.iloc[test_idx])
        for name, prediction in fold_predictions.items():
            predictions[name][test_idx] = prediction
        fold_ids[test_idx] = fold_id

    metrics = {name: regression_metrics(y, pred) for name, pred in predictions.items()}
    table = x.copy()
    table["actual"] = y.to_numpy()
    table["site_year_group"] = groups.to_numpy()
    table["outer_fold"] = fold_ids
    for name, pred in predictions.items():
        table[f"pred_{name}"] = pred
    return metrics, table


def raw_permutation_importance(
    predictor: Callable[[pd.DataFrame], np.ndarray],
    x: pd.DataFrame,
    y: pd.Series,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = mean_squared_error(y, predictor(x))
    rows = []
    for column in x.columns:
        increases = []
        for _ in range(n_repeats):
            permuted = x.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            increases.append(
                float(mean_squared_error(y, predictor(permuted)) - baseline)
            )
        rows.append(
            {
                "Feature": column,
                "Importance": float(np.mean(increases)),
                "Importance_SD": float(np.std(increases, ddof=1))
                if len(increases) > 1
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)


def plot_predictions(table: pd.DataFrame, target_label: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = {"RF": "#4C78A8", "XGBoost": "#F58518", "Stacking": "#54A24B"}
    lower = float(min(table["actual"].min(), table["pred_Stacking"].min()))
    upper = float(max(table["actual"].max(), table["pred_Stacking"].max()))

    for name in ["RF", "XGBoost", "Stacking"]:
        axes[0].scatter(
            table["actual"],
            table[f"pred_{name}"],
            s=12,
            alpha=0.48,
            label=name,
            color=colors[name],
        )
    axes[0].plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    axes[0].set_xlabel(f"Observed {target_label}")
    axes[0].set_ylabel(f"Predicted {target_label}")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    errors = [
        table[f"pred_{name}"] - table["actual"]
        for name in ["RF", "XGBoost", "Stacking"]
    ]
    axes[1].boxplot(errors, labels=["RF", "XGBoost", "Stacking"], showfliers=False)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Prediction error")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(f"{target_label}: predicted versus observed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, target_label: str, output_path: Path) -> None:
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(
        top["Feature"],
        top["Importance"],
        xerr=top["Importance_SD"],
        color="#4C78A8",
        alpha=0.85,
    )
    ax.set_xlabel("Increase in MSE after permutation")
    ax.set_ylabel("")
    ax.set_title(f"{target_label}: grouped permutation importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def run_target(args: argparse.Namespace, target_name: str) -> None:
    spec = TARGET_FILES[target_name]
    x_train, y_train, x_test, y_test = load_target(target_name)
    target_dir = args.output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    requested_modes = set(args.mode)
    if "all" in requested_modes:
        requested_modes = {"holdout", "record_cv", "grouped_cv"}

    metrics_summary = {}

    if "holdout" in requested_modes:
        print(f"[{spec['label']}] supplied 80/20 holdout", flush=True)
        holdout_metrics, holdout_predictions, holdout_models = evaluate_holdout(
            x_train,
            y_train,
            x_test,
            y_test,
            args.cv_folds,
            args.n_jobs,
            args.fast,
        )
        metrics_summary["holdout"] = holdout_metrics
        (target_dir / "metrics_holdout.json").write_text(
            json.dumps(holdout_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        holdout_predictions.to_csv(
            target_dir / "holdout_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        joblib.dump(holdout_models, target_dir / "holdout_models.joblib")
        plot_predictions(
            holdout_predictions,
            spec["label"],
            target_dir / "holdout_prediction_comparison.png",
        )

        def stacking_predict(frame):
            return predict_models(holdout_models, frame)["Stacking"]

        importance = raw_permutation_importance(
            stacking_predict,
            x_test,
            y_test,
            args.permutation_repeats,
            RANDOM_SEED,
        )
        importance.to_csv(
            target_dir / "holdout_permutation_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_importance(
            importance,
            spec["label"],
            target_dir / "holdout_permutation_importance.png",
        )

    x_all = pd.concat([x_train, x_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)

    for mode, label in [
        ("record_cv", "record CV"),
        ("grouped_cv", "site-year-grouped CV"),
    ]:
        if mode not in requested_modes:
            continue
        validation = "record" if mode == "record_cv" else "grouped"
        print(f"[{spec['label']}] {label}", flush=True)
        cv_metrics, cv_predictions = evaluate_cv(
            x_all,
            y_all,
            validation,
            args.cv_folds,
            args.n_jobs,
            args.fast,
            label,
        )
        metrics_summary[mode] = cv_metrics
        (target_dir / f"metrics_{mode}.json").write_text(
            json.dumps(cv_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cv_predictions.to_csv(
            target_dir / f"{mode}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_predictions(
            cv_predictions,
            spec["label"],
            target_dir / f"{mode}_prediction_comparison.png",
        )

    (target_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [
        {"validation": validation, "model": model_name, **metrics}
        for validation, model_metrics in metrics_summary.items()
        for model_name, metrics in model_metrics.items()
    ]
    pd.DataFrame(rows).to_csv(
        target_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_groups = set(make_group_key(x_train))
    test_groups = make_group_key(x_test)
    metadata = {
        "target": spec["label"],
        "requested_modes": sorted(requested_modes),
        "unit": spec["unit"],
        "random_seed": RANDOM_SEED,
        "cv_folds": args.cv_folds,
        "fast_grid": bool(args.fast),
        "supplied_split_rows": {"train": len(x_train), "test": len(x_test)},
        "supplied_split_site_year_groups": {
            "train": len(train_groups),
            "shared": len(train_groups & set(test_groups)),
            "test_record_share_sharing_site_year_with_train": float(
                test_groups.isin(train_groups).mean()
            ),
        },
        "categorical_columns": [col for col in CATEGORICAL_COLS if col in x_all.columns],
        "grouping_columns": GROUP_COLS,
        "preprocessing": {
            "numeric": "median imputation fitted on training data; StandardScaler fitted on training data",
            "categorical": "most-frequent imputation fitted on training data; one-hot encoding with unknown categories ignored",
        },
        "hyperparameter_selection": (
            "GridSearchCV with R2, fitted separately inside each outer training partition "
            "using inner five-fold splits of the same validation unit"
        ),
        "stacking": (
            "RF and XGBoost out-of-fold predictions form meta-features; Ridge alpha is selected "
            "only from training-partition out-of-fold predictions"
        ),
        "note": (
            "No constrained-scenario values are generated by this script. The repository is "
            "limited to reproducible model fitting and validation."
        ),
    }
    (target_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    for target_name in (list(TARGET_FILES) if args.target == "all" else [args.target]):
        run_target(args, target_name)


if __name__ == "__main__":
    main()
