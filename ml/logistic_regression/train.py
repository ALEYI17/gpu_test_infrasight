import cudf
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype


def main():
    print("📦 Loading dataset (fetch_covtype)...")
    X, y = fetch_covtype(return_X_y=True)
    print(f"number of samples: {X.shape[0]}, number of features: {X.shape[1]}")

    # Move data to GPU
    X = cudf.DataFrame(X)
    y = cudf.Series(y.astype("int32"))

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ⚙️ Standardize features on GPU
    print("📊 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🚀 Train cuML Logistic Regression model
    print("🚀 Training with cuML LogisticRegression on GPU...")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,            # regularization strength
        max_iter=1000,    # more iterations to ensure convergence
        tol=1e-4,
        solver="qn",
        verbose=True,
    )

    model.fit(X_train, y_train)

    # 🧠 Evaluate
    print("🧠 Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()

