"""
Decision Tree Classifier on scikit-learn Diabetes dataset (binarized at median)
Steps covered:
a) load dataset
b) split train/test
c) train DecisionTreeClassifier
d) predict
e) evaluate accuracy + confusion matrix (printed and saved as PNG)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


def main():
    # Load dataset
    data = load_diabetes()
    X = data.data
    y_reg = data.target  # continuous target (regression)

    # Convert to binary classes using the median as threshold
    median_target = np.median(y_reg)
    y = (y_reg > median_target).astype(int)  # 1 = above median progression

    # Train/test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Median of original (regression) target: {median_target:.2f}")
    print(f"Accuracy on test set: {acc:.4f}\n")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm, "\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["LowProg", "HighProg"]))

    # Plot & save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LowProg", "HighProg"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format='d')
    plt.title("Decision Tree Confusion Matrix (Diabetes as Binary)")
    plt.tight_layout()
    out_path = "confusion_matrix.png"
    plt.savefig(out_path, dpi=180)
    print(f"\nSaved confusion matrix figure to: {out_path}")

    # Show figure (works in most local VS Code setups)
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display figure: {e}")


if __name__ == "__main__":
    main()
