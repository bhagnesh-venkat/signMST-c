"""
Evaluate a trained checkpoint: greedy-decode the dev/test set and report
BLEU-4, ROUGE-L and WER -- the same metric families the paper uses
(Table 2/3), so your numbers are directly comparable in a report.

Usage:
    python -m src.evaluate --config configs/config.yaml \
        --checkpoint checkpoints/epoch30.pt --split test
"""
import functools
import json
from pathlib import Path

import sacrebleu
import torch
import yaml
from jiwer import wer
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PhoenixSLTDataset, collate_fn
from src.model import SLTTransformer
from src.vocab import Vocab, PAD, BOS, EOS


def main(cfg_path, checkpoint_path, split):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    vocab = Vocab.load(ckpt_dir / "vocab.json")

    d = cfg["data"]
    csv_key = {"dev": "dev_csv", "test": "test_csv"}[split]
    ds = PhoenixSLTDataset(
        csv_path=Path(d["root"]) / d[csv_key], frames_root=Path(d["root"]) / d["frames_root"],
        split=split, vocab=vocab, max_frames=d["max_frames"], frame_size=d["frame_size"],
        delimiter=d["csv_delimiter"],
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                         collate_fn=functools.partial(collate_fn, pad_id=vocab.stoi[PAD]))

    m = cfg["model"]
    model = SLTTransformer(
        vocab_size=len(vocab), pad_id=vocab.stoi[PAD], d_model=m["d_model"], nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"], num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
        cnn_backbone=m["cnn_backbone"], freeze_cnn=m["freeze_cnn"],
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    hyps, refs, names = [], [], []
    for batch in tqdm(loader, desc=f"decoding {split}"):
        frames = batch["frames"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        pred_ids = model.greedy_decode(frames, frame_mask, vocab.stoi[BOS], vocab.stoi[EOS])
        for i in range(pred_ids.size(0)):
            hyps.append(vocab.decode(pred_ids[i].tolist()))
        refs.extend(batch["translations"])
        names.extend(batch["names"])

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)) / len(refs)
    word_error_rate = wer(refs, hyps)

    print(f"\n== {split} set ({len(refs)} examples) ==")
    print(f"BLEU-4: {bleu.score:.2f}  ROUGE-L: {rouge_l * 100:.2f}  WER: {word_error_rate * 100:.2f}%")

    out_path = ckpt_dir / f"predictions_{split}.json"
    out_path.write_text(json.dumps(
        [{"name": n, "reference": r, "hypothesis": h} for n, r, h in zip(names, refs, hyps)],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("predictions saved to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split)
