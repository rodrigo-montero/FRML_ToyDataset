from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = Path("data")
METADATA_PATH = DATA_DIR / "metadata.csv"


def load_split(metadata, split):
    split_rows = metadata[metadata["split"] == split]

    X = []
    y = []

    for _, row in split_rows.iterrows():
        image_path = DATA_DIR / row["filename"]
        image = Image.open(image_path).convert("RGB")

        # Flatten image into one long feature vector
        x = np.array(image).reshape(-1) / 255.0

        X.append(x)
        y.append(row["label"])

    return np.array(X), np.array(y)


def evaluate(model, X, y, split_name):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y, predictions, target_names=["square", "circle"]))


def main():
    metadata = pd.read_csv(METADATA_PATH)

    X_train, y_train = load_split(metadata, "train")
    X_test_id, y_test_id = load_split(metadata, "test_id")
    X_test_ood, y_test_ood = load_split(metadata, "test_ood")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Tiny shortcut-learning experiment")
    print("--------------------------------")
    print("Training data:")
    print("- square = blue")
    print("- circle = red")
    print()
    print("OOD test data:")
    print("- square = red")
    print("- circle = blue")

    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_test_id, y_test_id, "Test ID")
    evaluate(model, X_test_ood, y_test_ood, "Test OOD")


if __name__ == "__main__":
    main()