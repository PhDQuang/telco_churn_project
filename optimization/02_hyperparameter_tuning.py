from pathlib import Path
import json
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# =====================================================
# 1. PATH CONFIG
# =====================================================
def get_project_root() -> Path:
    """
    File này nằm ở: optimization/hyperparameter_tuning.py
    => project root là thư mục cha của optimization
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()

DATA_FE_ROOT = PROJECT_ROOT / "data" / "feature_engineering"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hyperparameter_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_FE_ROOT:", DATA_FE_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)


# =====================================================
# 2. FILE MAP
# =====================================================
DATASET_CONFIG = {
    "logistic_regression": {
        "file": DATA_FE_ROOT / "logistic_regression" / "telco_churn_fe_logistic.csv",
        "model_name": "logistic_regression"
    },
    "naive_bayes": {
        "file": DATA_FE_ROOT / "naive_bayes" / "telco_churn_fe_nb.csv",
        "model_name": "gaussian_naive_bayes"
    },
    "decision_tree": {
        "file": DATA_FE_ROOT / "decision_tree" / "telco_churn_fe_tree.csv",
        "model_name": "decision_tree"
    },
    "random_forest": {
        "file": DATA_FE_ROOT / "random_forest" / "telco_churn_fe_rf.csv",
        "model_name": "random_forest"
    }
}


# =====================================================
# 3. COMMON HELPERS
# =====================================================
def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Churn" not in df.columns:
        raise ValueError(f"File {csv_path.name} không có cột 'Churn'")

    df = df.copy()

    # Chuẩn hóa target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        raise ValueError(f"Cột Churn trong {csv_path.name} có giá trị ngoài Yes/No")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return df, X, y


def detect_feature_types(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return categorical_cols, numeric_cols


def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }
    return result


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =====================================================
# 4. PREPROCESSORS
# =====================================================
def build_preprocessor_for_logistic(X: pd.DataFrame):
    categorical_cols, numeric_cols = detect_feature_types(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def build_preprocessor_for_nb(X: pd.DataFrame):
    """
    GaussianNB không làm việc trực tiếp tốt với sparse matrix,
    nên OneHotEncoder dùng sparse_output=False để trả về dense matrix.
    Đồng thời dùng MinMaxScaler để numeric gọn hơn.
    """
    categorical_cols, numeric_cols = detect_feature_types(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop",
        sparse_threshold=0
    )
    return preprocessor


def build_preprocessor_for_tree_rf(X: pd.DataFrame):
    categorical_cols, numeric_cols = detect_feature_types(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


# =====================================================
# 5. TRAIN / TUNE EACH MODEL
# =====================================================
def tune_logistic_regression(X_train, y_train):
    preprocessor = build_preprocessor_for_logistic(X_train)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=3000, random_state=42))
    ])

    param_grid = [
        {
            "model__solver": ["liblinear"],
            "model__penalty": ["l1", "l2"],
            "model__C": [0.01, 0.1, 1, 10, 50],
            "model__class_weight": [None, "balanced"]
        },
        {
            "model__solver": ["lbfgs"],
            "model__penalty": ["l2"],
            "model__C": [0.01, 0.1, 1, 10, 50],
            "model__class_weight": [None, "balanced"]
        }
    ]

    return pipeline, param_grid


def tune_gaussian_nb(X_train, y_train):
    preprocessor = build_preprocessor_for_nb(X_train)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GaussianNB())
    ])

    param_grid = {
        "model__var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    }

    return pipeline, param_grid


def tune_decision_tree(X_train, y_train):
    preprocessor = build_preprocessor_for_tree_rf(X_train)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [3, 5, 7, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": [None, "sqrt", "log2"],
        "model__class_weight": [None, "balanced"]
    }

    return pipeline, param_grid


def tune_random_forest(X_train, y_train):
    preprocessor = build_preprocessor_for_tree_rf(X_train)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced"]
    }

    return pipeline, param_grid


# =====================================================
# 6. GRID SEARCH RUNNER
# =====================================================
def run_grid_search(X_train, y_train, pipeline, param_grid, scoring="f1"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    grid.fit(X_train, y_train)
    return grid


# =====================================================
# 7. MAIN PIPELINE FOR ONE ALGORITHM
# =====================================================
def run_one_algorithm(algorithm_key: str):
    config = DATASET_CONFIG[algorithm_key]
    csv_path = config["file"]
    model_name = config["model_name"]

    print("\n" + "=" * 80)
    print(f"RUNNING: {algorithm_key}")
    print("DATA FILE:", csv_path)

    df, X, y = load_dataset(csv_path)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if algorithm_key == "logistic_regression":
        pipeline, param_grid = tune_logistic_regression(X_train, y_train)
        scoring = "f1"

    elif algorithm_key == "naive_bayes":
        pipeline, param_grid = tune_gaussian_nb(X_train, y_train)
        scoring = "f1"

    elif algorithm_key == "decision_tree":
        pipeline, param_grid = tune_decision_tree(X_train, y_train)
        scoring = "f1"

    elif algorithm_key == "random_forest":
        pipeline, param_grid = tune_random_forest(X_train, y_train)
        scoring = "f1"

    else:
        raise ValueError(f"Thuật toán không hợp lệ: {algorithm_key}")

    grid = run_grid_search(X_train, y_train, pipeline, param_grid, scoring=scoring)

    best_model = grid.best_estimator_
    eval_result = evaluate_model(best_model, X_test, y_test, model_name)

    result_summary = {
        "algorithm_key": algorithm_key,
        "model_name": model_name,
        "data_file": str(csv_path),
        "best_score_cv": float(grid.best_score_),
        "best_params": grid.best_params_,
        "test_metrics": {
            "accuracy": eval_result["accuracy"],
            "precision": eval_result["precision"],
            "recall": eval_result["recall"],
            "f1_score": eval_result["f1_score"]
        },
        "confusion_matrix": eval_result["confusion_matrix"],
        "classification_report": eval_result["classification_report"]
    }

    save_path = OUTPUT_DIR / f"{algorithm_key}_tuning_result.json"
    save_json(result_summary, save_path)

    print("\nBest CV Score:", grid.best_score_)
    print("Best Params:", grid.best_params_)
    print("\nTest Metrics:")
    print(" - Accuracy :", eval_result["accuracy"])
    print(" - Precision:", eval_result["precision"])
    print(" - Recall   :", eval_result["recall"])
    print(" - F1-score :", eval_result["f1_score"])
    print("\nSaved to:", save_path)

    return result_summary


# =====================================================
# 8. RUN ALL + SAVE SUMMARY CSV
# =====================================================
def main():
    all_results = []

    for algorithm_key in DATASET_CONFIG.keys():
        try:
            result = run_one_algorithm(algorithm_key)
            all_results.append({
                "algorithm_key": result["algorithm_key"],
                "model_name": result["model_name"],
                "best_score_cv": result["best_score_cv"],
                "accuracy": result["test_metrics"]["accuracy"],
                "precision": result["test_metrics"]["precision"],
                "recall": result["test_metrics"]["recall"],
                "f1_score": result["test_metrics"]["f1_score"],
                "best_params": json.dumps(result["best_params"], ensure_ascii=False)
            })
        except Exception as e:
            print(f"\nLỗi khi chạy {algorithm_key}: {e}")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = OUTPUT_DIR / "hyperparameter_tuning_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 80)
        print("Đã lưu bảng tổng hợp:", summary_path)
        print(summary_df.sort_values(by="f1_score", ascending=False))
    else:
        print("\nKhông có kết quả nào được tạo.")


if __name__ == "__main__":
    main()