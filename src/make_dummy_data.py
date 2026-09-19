"""
Creates a TINY fake dataset with the exact same folder/file structure as
PHOENIX-2014-T, but filled with random noise images and made-up sentences.

Why this exists: getting real access to PHOENIX-2014-T can take days/weeks.
This script lets you prove your code actually runs -- loads data, trains,
saves a checkpoint, evaluates -- while you wait for real access. It is NOT
meant to produce a good model, only to prove the pipeline works.

Usage:
    python -m src.make_dummy_data
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/phoenix2014T")
SENTENCES = [
    "es wird sonnig",
    "morgen regnet es",
    "der wind kommt aus nordwesten",
    "am wochenende wird es kalt",
    "die temperatur steigt leicht",
    "im sueden gibt es gewitter",
    "am montag scheint die sonne",
    "es bleibt bewoelkt",
]


def make_video_folder(folder: Path, n_frames: int = 8):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"frame_{i:03d}.png")


def make_split(split: str, n_examples: int):
    rows = []
    frames_dir = ROOT / "features/fullFrame-210x260px" / split
    for i in range(n_examples):
        name = f"{split}_{i:03d}"
        make_video_folder(frames_dir / name)
        rows.append({
            "name": name, "video": name, "start": 0, "end": 8,
            "speaker": "signer1", "orth": "DUMMY GLOSS",
            "translation": SENTENCES[i % len(SENTENCES)],
        })
    df = pd.DataFrame(rows)
    out_csv = ROOT / "annotations" / f"PHOENIX-2014-T.{split}.corpus.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, sep="|", index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    make_split("train", 6)
    make_split("dev", 2)
    make_split("test", 2)
    print("\nDone. This is FAKE data, only for testing that the code runs.")
    print("Next: python -m src.inspect_data --config configs/config.yaml")
