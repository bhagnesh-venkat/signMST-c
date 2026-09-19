"""
Baseline Sign-to-Text model: per-frame CNN -> Transformer encoder-decoder.

This is a simplified version of the paper's SignMST-C "Multimodal Module for
SLT" (Sec. 4, Fig. 3), using ONLY the video stream for now. Landmark fusion,
the self-supervised rapid-motion pretraining, and the distillation losses
(Eq. 7-9) are Phase 3/4 in docs/ROADMAP.md -- add them once this baseline
trains and produces sane output.
"""
import math

import torch
import torch.nn as nn
from torchvision import models


class FrameEncoder(nn.Module):
    """Per-frame CNN feature extractor (2D ResNet, pretrained on ImageNet)."""

    def __init__(self, d_model, backbone="resnet18", freeze=True):
        super().__init__()
        net = getattr(models, backbone)(weights="DEFAULT")
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.cnn = net
        self.proj = nn.Linear(feat_dim, d_model)
        if freeze:
            for p in self.cnn.parameters():
                p.requires_grad = False

    def forward(self, frames):
        # frames: (B, T, 3, H, W)
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        feats = self.cnn(x)               # (B*T, feat_dim)
        feats = self.proj(feats)          # (B*T, d_model)
        return feats.view(B, T, -1)       # (B, T, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SLTTransformer(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1,
                 cnn_backbone="resnet18", freeze_cnn=True):
        super().__init__()
        self.pad_id = pad_id
        self.frame_encoder = FrameEncoder(d_model, cnn_backbone, freeze_cnn)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(size, device):
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def forward(self, frames, frame_mask, tgt_in):
        # frames: (B, T, 3, H, W); frame_mask: (B, T) True=pad
        # tgt_in: (B, L) shifted-right target ids
        src = self.pos_enc(self.frame_encoder(frames))
        tgt = self.tgt_pos_enc(self.tok_emb(tgt_in))

        tgt_mask = self.causal_mask(tgt_in.size(1), tgt_in.device)
        tgt_pad_mask = tgt_in.eq(self.pad_id)

        out = self.transformer(
            src, tgt,
            src_key_padding_mask=frame_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=frame_mask,
        )
        return self.generator(out)  # (B, L, vocab_size)

    @torch.no_grad()
    def greedy_decode(self, frames, frame_mask, bos_id, eos_id, max_len=60):
        device = frames.device
        B = frames.size(0)
        src = self.pos_enc(self.frame_encoder(frames))
        memory = self.transformer.encoder(src, src_key_padding_mask=frame_mask)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt = self.tgt_pos_enc(self.tok_emb(ys))
            tgt_mask = self.causal_mask(ys.size(1), device)
            out = self.transformer.decoder(
                tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=frame_mask
            )
            logits = self.generator(out[:, -1])
            next_tok = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= next_tok.squeeze(1).eq(eos_id)
            if finished.all():
                break
        return ys
