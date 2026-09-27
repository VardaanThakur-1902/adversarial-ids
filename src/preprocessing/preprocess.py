from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_dataset():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    dataframes = []

    for file in csv_files:
        print(f"Loading: {file.name}")

        df = pd.read_csv(file)

        # Remove spaces around column names
        df.columns = df.columns.str.strip()

        dataframes.append(df)

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(f"\nInitial shape: {df.shape}")

    return df


# --------------------------------------------------
# Clean dataset
# --------------------------------------------------

def clean_dataset(df):

    # Clean labels
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Replace infinity values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows containing NaN
    before = len(df)

    df = df.dropna()

    after = len(df)

    print(f"Rows removed because of NaN/inf: {before - after}")

    # Remove duplicate rows
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Duplicate rows removed: {before - after}")

    # Create binary target
    df["target"] = (
        df["Label"]
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    return df


# --------------------------------------------------
# Prepare features
# --------------------------------------------------

def prepare_features(df):

    X = df.drop(
        columns=["Label", "target"]
    )

    y = df["target"]

    labels = df["Label"]

    # Find non-numeric features
    non_numeric = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        print(
            "WARNING - Non-numeric features:",
            non_numeric
        )

    # Remove constant features
    constant_features = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    if constant_features:
        print(
            "Removing constant features:",
            constant_features
        )

        X = X.drop(
            columns=constant_features
        )

    print(
        f"Number of features: {X.shape[1]}"
    )

    return X, y, labels


# --------------------------------------------------
# Split dataset
# --------------------------------------------------

def split_dataset(X, y, labels):

    X_train, X_temp, y_train, y_temp, label_train, label_temp = train_test_split(
        X,
        y,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test, label_val, label_test = train_test_split(
        X_temp,
        y_temp,
        label_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_train,
        label_val,
        label_test
    )


# --------------------------------------------------
# Scale dataset
# --------------------------------------------------

def scale_features(
    X_train,
    X_val,
    X_test
):

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit only on training data
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        X_val_scaled,
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

def save_splits(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    label_train,
    label_val,
    label_test,
    scaler
):

    X_train.to_csv(
        SPLIT_DIR / "X_train.csv",
        index=False
    )

    X_val.to_csv(
        SPLIT_DIR / "X_val.csv",
        index=False
    )

    X_test.to_csv(
        SPLIT_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        SPLIT_DIR / "y_train.csv",
        index=False
    )

    y_val.to_csv(
        SPLIT_DIR / "y_val.csv",
        index=False
    )

    y_test.to_csv(
        SPLIT_DIR / "y_test.csv",
        index=False
    )

    label_train.to_csv(
        SPLIT_DIR / "label_train.csv",
        index=False
    )

    label_val.to_csv(
        SPLIT_DIR / "label_val.csv",
        index=False
    )

    label_test.to_csv(
        SPLIT_DIR / "label_test.csv",
        index=False
    )

    # Save scaler
    import joblib

    joblib.dump(
        scaler,
        SPLIT_DIR / "scaler.pkl"
    )

    print("\nProcessed datasets saved.")
    print("Scaler saved.")


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    df = load_dataset()

    df = clean_dataset(df)

    print(
        f"Shape after cleaning: {df.shape}"
    )

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    X, y, labels = prepare_features(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_train,
        label_val,
        label_test
    ) = split_dataset(
        X,
        y,
        labels
    )

    print("\nSplit sizes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    print("\nScaled datasets:")
    print("Train:", X_train_scaled.shape)
    print("Validation:", X_val_scaled.shape)
    print("Test:", X_test_scaled.shape)

    save_splits(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        label_train,
        label_val,
        label_test,
        scaler
    )

if __name__ == "__main__":
    main()