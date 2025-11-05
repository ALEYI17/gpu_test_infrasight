import cudf
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
from sklearn.datasets import fetch_covtype


def main():
    print("📦 Loading dataset (fetch_covtype)...")
    X, y = fetch_covtype(return_X_y=True)
    print(f"number of samples: {X.shape[0]}, number of features: {X.shape[1]}")

    # Move to GPU
    X = cudf.DataFrame(X)
    y = cudf.Series(y.astype("int32"))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (optional, but good practice)
    print("📊 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🚀 Train GPU Random Forest
    print("🚀 Training cuML RandomForest on GPU...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        n_streams=8,
        verbose=True
    )
    rf.fit(X_train, y_train)

    # Evaluate
    print("🧠 Evaluating Random Forest...")
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()

