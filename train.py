import os
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import load_data, preprocess_data

def main():
    # Load dataset
    data_path = os.path.join("data", "fake_or_real_news.csv")
    df = load_data(data_path)

    # Preprocess
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    # Train model
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {score:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
