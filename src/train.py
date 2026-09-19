"""
Train the baseline SLT model.

Usage (from repo root):
    python -m src.train --config configs/config.yaml
"""
import argparse
import functools
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PhoenixSLTDataset, collate_fn
from src.model import SLTTransformer
from src.utils import set_seed, save_checkpoint
from src.vocab import Vocab, PAD


def build_dataloaders(cfg, vocab):
    d = cfg["data"]
    datasets = {}
    for split, csv_key in [("train", "train_csv"), ("dev", "dev_csv")]:
        datasets[split] = PhoenixSLTDataset(
            csv_path=Path(d["root"]) / d[csv_key],
            frames_root=Path(d["root"]) / d["frames_root"],
            split=split, vocab=vocab, max_frames=d["max_frames"],
            frame_size=d["frame_size"], delimiter=d["csv_delimiter"],
        )
    pad_id = vocab.stoi[PAD]
    loaders = {
        split: DataLoader(
            ds, batch_size=cfg["train"]["batch_size"], shuffle=(split == "train"),
            num_workers=cfg["train"]["num_workers"],
            collate_fn=functools.partial(collate_fn, pad_id=pad_id),
        )
        for split, ds in datasets.items()
    }
    return loaders


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        frames = batch["frames"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        targets = batch["targets"].to(device)
        tgt_in, tgt_out = targets[:, :-1], targets[:, 1:]
        logits = model(frames, frame_mask, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = cfg["data"]
    train_csv = Path(d["root"]) / d["train_csv"]
    translations = pd.read_csv(train_csv, sep=d["csv_delimiter"])["translation"]
    vocab = Vocab.build(translations, min_freq=cfg["vocab"]["min_freq"],
                         lowercase=cfg["vocab"]["lowercase"])
    print("vocab size:", len(vocab))

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(ckpt_dir / "vocab.json")

    loaders = build_dataloaders(cfg, vocab)

    m = cfg["model"]
    model = SLTTransformer(
        vocab_size=len(vocab), pad_id=vocab.stoi[PAD],
        d_model=m["d_model"], nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"], num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
        cnn_backbone=m["cnn_backbone"], freeze_cnn=m["freeze_cnn"],
    ).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg["train"]["lr"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD], label_smoothing=0.1)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            targets = batch["targets"].to(device)

            tgt_in, tgt_out = targets[:, :-1], targets[:, 1:]
            logits = model(frames, frame_mask, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if step % cfg["train"]["log_every"] == 0:
                pbar.set_postfix(loss=total_loss / n_batches)

        val_loss = evaluate_loss(model, loaders["dev"], criterion, device)
        print(f"epoch {epoch}: train_loss={total_loss / n_batches:.4f} val_loss={val_loss:.4f}")

        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, ckpt_dir / f"epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
