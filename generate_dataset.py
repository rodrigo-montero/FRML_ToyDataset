from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import random

IMG_SIZE = 32
N_TRAIN = 1000
N_TEST = 200

OUTPUT_DIR = Path("data")

COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
}

BACKGROUND = (255, 255, 255)

LABEL_TO_SHAPE = {
    0: "square",
    1: "circle",
}

def shortcut_color(label, split):
    if split in ["train", "test_id"]:
        return "blue" if label == 0 else "red"
    elif split == "test_ood":
        return "red" if label == 0 else "blue"
    else:
        raise ValueError(f"Unknown split: {split}")

def draw_shape(shape, color):
    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)

    margin = 8
    box = [margin, margin, IMG_SIZE - margin, IMG_SIZE - margin]

    if shape == "square":
        draw.rectangle(box, fill=COLORS[color])
    elif shape == "circle":
        draw.ellipse(box, fill=COLORS[color])
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return image

def generate_split(split, n_samples):
    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in range(n_samples):
        label = random.choice([0, 1])
        shape = LABEL_TO_SHAPE[label]
        color = shortcut_color(label, split)

        image = draw_shape(shape, color)

        filename = f"{split}_{i:04d}.png"
        image_path = split_dir / filename
        image.save(image_path)

        rows.append({
            "filename": str(Path(split) / filename),
            "split": split,
            "label": label,
            "shape": shape,
            "color": color,
            "intended_feature": "shape",
            "shortcut_feature": "color",
        })

    return rows

def main():
    random.seed(42)

    all_rows = []
    all_rows += generate_split("train", N_TRAIN)
    all_rows += generate_split("test_id", N_TEST)
    all_rows += generate_split("test_ood", N_TEST)

    metadata = pd.DataFrame(all_rows)
    metadata.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    print("Dataset generated.")
    print(metadata.groupby(["split", "shape", "color", "label"]).size())

if __name__ == "__main__":
    main()