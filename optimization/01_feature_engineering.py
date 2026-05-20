# ============================================
# FEATURE ENGINEERING FOR 4 ALGORITHMS
# SAVE TO: data/feature_engineering/<algorithm>/
# ============================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# 1. PATH CONFIG
# -------------------------------------------------
def get_project_root() -> Path:
    """
    Xác định thư mục gốc của project.
    File này đang nằm ở: optimization/feature_engineering.py
    => project root là thư mục cha của optimization
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FE_ROOT = PROJECT_ROOT / "data" / "feature_engineering"

CLEAN_FILE = DATA_PROCESSED_DIR / "telco_churn_clean.csv"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CLEAN_FILE:", CLEAN_FILE)
print("DATA_FE_ROOT:", DATA_FE_ROOT)
print("CLEAN_FILE exists:", CLEAN_FILE.exists())


# -------------------------------------------------
# 2. COMMON FEATURE ENGINEERING
# -------------------------------------------------
def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các feature chung dùng cho nhiều thuật toán.
    Giả định file clean vẫn giữ các cột categorical dạng text như:
    - Contract
    - PaymentMethod
    - OnlineSecurity
    - StreamingTV
    - ...
    """
    df = df.copy()

    required_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc trong file clean: {missing_cols}")

    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    def service_to_binary(x):
        return 1 if x == "Yes" else 0

    for col in addon_cols:
        df[f"{col}_bin"] = df[col].apply(service_to_binary)

    # Tổng số dịch vụ cộng thêm
    df["NumAddOnServices"] = df[[f"{col}_bin" for col in addon_cols]].sum(axis=1)

    # Có bảo mật / hỗ trợ kỹ thuật
    df["HasSecuritySupport"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    # Có dùng dịch vụ streaming
    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # Hợp đồng theo tháng / dài hạn
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsLongTermContract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    # Phương thức thanh toán
    df["IsAutoPayment"] = df["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # Chi phí trung bình theo tháng
    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Độ lệch giữa tổng tiền và ước lượng
    df["ChargeGap"] = df["TotalCharges"] - (df["tenure"] * df["MonthlyCharges"])

    # Nhóm thời gian gắn bó
    def tenure_group(t):
        if t <= 12:
            return "0_12"
        elif t <= 24:
            return "13_24"
        elif t <= 48:
            return "25_48"
        else:
            return "49_plus"

    df["TenureGroup"] = df["tenure"].apply(tenure_group)

    return df


# -------------------------------------------------
# 3. FEATURE ENGINEERING FOR EACH ALGORITHM
# -------------------------------------------------
def feature_engineering_for_logistic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic Regression:
    - giữ feature chung
    - thêm interaction vì logistic là mô hình tuyến tính
    """
    df = add_common_features(df)

    df["MonthlyCharges_IsMonthToMonth"] = df["MonthlyCharges"] * df["IsMonthToMonth"]
    df["MonthlyCharges_SeniorCitizen"] = df["MonthlyCharges"] * df["SeniorCitizen"]
    df["Tenure_MonthlyCharges"] = df["tenure"] * df["MonthlyCharges"]

    return df


def feature_engineering_for_naive_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive Bayes:
    - giữ feature đơn giản hơn
    - hạn chế quá nhiều interaction
    - bỏ bớt các cột _bin để giảm dư thừa
    """
    df = add_common_features(df)

    drop_cols = [
        "OnlineSecurity_bin",
        "OnlineBackup_bin",
        "DeviceProtection_bin",
        "TechSupport_bin",
        "StreamingTV_bin",
        "StreamingMovies_bin"
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    return df


def feature_engineering_for_decision_tree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision Tree:
    - hợp với feature dạng cờ / ngưỡng / nhóm
    """
    df = add_common_features(df)

    df["TenureFlag_Short"] = (df["tenure"] <= 12).astype(int)
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] >= df["MonthlyCharges"].median()).astype(int)
    df["HighTotalCharges"] = (df["TotalCharges"] >= df["TotalCharges"].median()).astype(int)

    return df


def feature_engineering_for_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Random Forest:
    - giữ feature chung
    - thêm cả interaction + feature nhóm / cờ
    """
    df = add_common_features(df)

    df["MonthlyCharges_IsMonthToMonth"] = df["MonthlyCharges"] * df["IsMonthToMonth"]
    df["MonthlyCharges_SeniorCitizen"] = df["MonthlyCharges"] * df["SeniorCitizen"]
    df["Tenure_MonthlyCharges"] = df["tenure"] * df["MonthlyCharges"]

    df["TenureFlag_Short"] = (df["tenure"] <= 12).astype(int)
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] >= df["MonthlyCharges"].median()).astype(int)
    df["HighTotalCharges"] = (df["TotalCharges"] >= df["TotalCharges"].median()).astype(int)

    return df


# -------------------------------------------------
# 4. SAVE FUNCTION
# -------------------------------------------------
def save_fe_version(df: pd.DataFrame, algorithm_name: str, prefix_name: str) -> None:
    """
    Lưu:
    - full dataframe
    - X_train, X_test, y_train, y_test
    vào data/feature_engineering/<algorithm_name>/
    """
    if "Churn" not in df.columns:
        raise ValueError("Cột 'Churn' không tồn tại trong dataframe.")

    save_dir = DATA_FE_ROOT / algorithm_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Lưu full dataframe
    full_path = save_dir / f"telco_churn_{prefix_name}.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")

    # Tách X, y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_path = save_dir / f"X_train_{prefix_name}.csv"
    X_test_path = save_dir / f"X_test_{prefix_name}.csv"
    y_train_path = save_dir / f"y_train_{prefix_name}.csv"
    y_test_path = save_dir / f"y_test_{prefix_name}.csv"

    X_train.to_csv(X_train_path, index=False, encoding="utf-8-sig")
    X_test.to_csv(X_test_path, index=False, encoding="utf-8-sig")
    y_train.to_csv(y_train_path, index=False, encoding="utf-8-sig")
    y_test.to_csv(y_test_path, index=False, encoding="utf-8-sig")

    print(f"\nĐã lưu cho {algorithm_name}:")
    print(" -", full_path)
    print(" -", X_train_path)
    print(" -", X_test_path)
    print(" -", y_train_path)
    print(" -", y_test_path)


# -------------------------------------------------
# 5. MAIN RUN
# -------------------------------------------------
def main():
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file clean tại: {CLEAN_FILE}\n"
            f"Hãy kiểm tra lại notebook/data cleaning hoặc tên file."
        )

    # Đọc data đã clean
    df_base = pd.read_csv(CLEAN_FILE)

    if "Churn" not in df_base.columns:
        raise ValueError("File clean phải chứa cột 'Churn'.")

    # Tạo 4 phiên bản FE
    df_logistic = feature_engineering_for_logistic(df_base)
    df_nb = feature_engineering_for_naive_bayes(df_base)
    df_tree = feature_engineering_for_decision_tree(df_base)
    df_rf = feature_engineering_for_random_forest(df_base)

    # Lưu
    save_fe_version(df_logistic, "logistic_regression", "fe_logistic")
    save_fe_version(df_nb, "naive_bayes", "fe_nb")
    save_fe_version(df_tree, "decision_tree", "fe_tree")
    save_fe_version(df_rf, "random_forest", "fe_rf")

    print("\nHoàn tất lưu 4 phiên bản feature engineering.")


if __name__ == "__main__":
    main()