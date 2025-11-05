import cudf
from cuml.svm import SVC
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

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize
    print("📊 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🚀 Train GPU SVM
    print("🚀 Training cuML SVM on GPU...")
    svm = SVC(
        C=1.0,
        kernel="rbf",   # try 'linear' or 'poly' too
        gamma="scale",
        max_iter=1000,
        verbose=True
    )

    svm.fit(X_train, y_train)

    # Evaluate
    print("🧠 Evaluating SVM...")
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()

