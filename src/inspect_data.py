"""
Run this FIRST, before training, to sanity-check that your PHOENIX-2014-T
download matches the paths/column names assumed in configs/config.yaml.
Dataset releases occasionally differ slightly in folder/column naming, so
this script is your safety net -- fix the config until everything here
prints sensible values.

    python -m src.inspect_data --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    d = cfg["data"]
    root = Path(d["root"])

    for split, key in [("train", "train_csv"), ("dev", "dev_csv"), ("test", "test_csv")]:
        csv_path = root / d[key]
        print(f"\n--- {split}: {csv_path} ---")
        if not csv_path.exists():
            print("  NOT FOUND. Check `data.root` / `data.<split>_csv` in the config.")
            continue
        df = pd.read_csv(csv_path, sep=d["csv_delimiter"])
        print("  columns:", list(df.columns))
        print("  rows:", len(df))
        print(df.head(2).to_string())

        frames_root = root / d["frames_root"] / split
        if "video" in df.columns:
            sample_dir = frames_root / df.iloc[0]["video"]
            exists = sample_dir.is_dir()
            n_frames = len(list(sample_dir.glob("*.png"))) if exists else 0
            print(f"  sample frame folder exists: {exists} ({n_frames} frames) -> {sample_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
