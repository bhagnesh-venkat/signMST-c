"""
PyTorch Dataset for RWTH-PHOENIX-2014-T.

Expected layout after you extract the official release (folder names vary
slightly between release versions -- ALWAYS run `python -m src.inspect_data`
first and fix configs/config.yaml to match what you actually downloaded):

data/phoenix2014T/
|-- annotations/manual/
|   |-- PHOENIX-2014-T.train.corpus.csv
|   |-- PHOENIX-2014-T.dev.corpus.csv
|   `-- PHOENIX-2014-T.test.corpus.csv
`-- features/fullFrame-210x260px/
    |-- train/<video_folder>/*.png
    |-- dev/<video_folder>/*.png
    `-- test/<video_folder>/*.png

The annotation CSVs are '|'-delimited with (at least) these columns:
    name | video | start | end | speaker | orth | translation
`video` holds the relative path to the folder of frame images, and
`translation` is the German sentence we want to predict.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_video_dir(video_field: str) -> str:
    """
    The official corpus CSV's `video` column is sometimes a plain folder
    name, but in the v3 release it's a glob pattern like
    '<name>/1/*.png' (the '1' is a camera-angle subfolder, and '*.png'
    matches every frame in it). This strips that down to the actual
    folder path so we can list the folder ourselves.
    """
    if "*" in video_field:
        return str(Path(video_field).parent)
    return video_field


def uniform_subsample(items, max_len):
    """Evenly pick at most max_len items, preserving order."""
    if len(items) <= max_len:
        return items
    idx = np.linspace(0, len(items) - 1, max_len).round().astype(int)
    return [items[i] for i in idx]


class PhoenixSLTDataset(Dataset):
    def __init__(self, csv_path, frames_root, split, vocab, max_frames=150,
                 frame_size=224, delimiter="|"):
        self.df = pd.read_csv(csv_path, sep=delimiter)
        self.frames_root = Path(frames_root) / split
        self.vocab = vocab
        self.max_frames = max_frames

        self.transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # sanity check: only keep rows whose frame folder actually exists on disk
        keep = []
        for i, row in self.df.iterrows():
            video_dir = resolve_video_dir(row["video"])
            if (self.frames_root / video_dir).is_dir():
                keep.append(i)
        dropped = len(self.df) - len(keep)
        if dropped:
            print(f"[{split}] warning: {dropped} rows skipped (frame folder not found)")
        self.df = self.df.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_frames(self, video_dir: Path):
        paths = sorted(video_dir.glob("*.png")) or sorted(video_dir.glob("*.jpg"))
        paths = uniform_subsample(paths, self.max_frames)
        frames = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(frames)  # (T, 3, H, W)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_dir = resolve_video_dir(row["video"])
        frames = self._load_frames(self.frames_root / video_dir)
        target_ids = torch.tensor(self.vocab.encode(row["translation"]), dtype=torch.long)
        return {
            "name": row["name"],
            "frames": frames,                 # (T, 3, H, W)
            "target": target_ids,             # (L,)
            "translation": row["translation"],
        }


def collate_fn(batch, pad_id: int):
    """Pads a batch of variable-length videos/sentences to the same size."""
    max_t = max(b["frames"].shape[0] for b in batch)
    max_l = max(b["target"].shape[0] for b in batch)
    B = len(batch)
    _, C, H, W = batch[0]["frames"].shape

    frames = torch.zeros(B, max_t, C, H, W)
    frame_mask = torch.ones(B, max_t, dtype=torch.bool)  # True = PAD (ignored by attention)
    targets = torch.full((B, max_l), pad_id, dtype=torch.long)
    names, translations = [], []

    for i, b in enumerate(batch):
        t = b["frames"].shape[0]
        l = b["target"].shape[0]
        frames[i, :t] = b["frames"]
        frame_mask[i, :t] = False
        targets[i, :l] = b["target"]
        names.append(b["name"])
        translations.append(b["translation"])

    return {
        "frames": frames, "frame_mask": frame_mask,
        "targets": targets, "names": names, "translations": translations,
    }