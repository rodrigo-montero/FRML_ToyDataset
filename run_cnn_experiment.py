from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("data")
METADATA_PATH = DATA_DIR / "metadata.csv"

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
SEED = 42


class ShapeDataset(Dataset):
    def __init__(self, metadata, split):
        self.rows = metadata[metadata["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        image_path = DATA_DIR / row["filename"]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0

        # Convert from H x W x C to C x H x W
        image = torch.tensor(image).permute(2, 0, 1)

        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0

    for images, labels in loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, split_name):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"{split_name} Accuracy: {accuracy:.3f}")

    return accuracy


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    metadata = pd.read_csv(METADATA_PATH)

    train_dataset = ShapeDataset(metadata, "train")
    test_id_dataset = ShapeDataset(metadata, "test_id")
    test_ood_dataset = ShapeDataset(metadata, "test_ood")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_id_loader = DataLoader(test_id_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("CNN shortcut-learning experiment")
    print("--------------------------------")

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

    print()
    evaluate(model, train_loader, "Train")
    evaluate(model, test_id_loader, "Test ID")
    evaluate(model, test_ood_loader, "Test OOD")


if __name__ == "__main__":
    main()