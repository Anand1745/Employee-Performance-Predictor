# src/evaluate.py

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }

def print_evaluation(results):
    print(f"\n✅ Accuracy: {results['accuracy']:.2f}")
    print("\nConfusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['report'])