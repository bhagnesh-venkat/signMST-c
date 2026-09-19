# Implementation Roadmap — mapped to the paper

Paper: *"Beyond Words: AuralLLM and SignMST-C for Precise Sign Language
Production and Bidirectional Accessibility"* (Li, Zhang, et al.). This
roadmap covers the **sign-to-text (SLT) half only** — i.e. SignMST-C
(paper Section 4, "SLT Framework", Figure 3).

| Phase | What it is | Paper reference | Status |
|---|---|---|---|
| 1 | Data pipeline: PHOENIX-2014-T loader, uniform frame sampling, word-level vocabulary | Sec. 3 (dataset benchmarking) | ✅ Done |
| 2 | **Baseline model**: single-stream 2D-CNN + Transformer encoder-decoder, trained end-to-end, evaluated with BLEU-4 / ROUGE-L / WER | Closest to the classic SLT baselines in Table 2 (e.g. Joint-SLRT) | ✅ Done |
| 3 | Multimodal fusion: add a MediaPipe landmark stream (1D conv over keypoints), concatenate with video features before the Transformer, add the three KL-divergence distillation losses | Sec. 4, "Multimodal Module for SLT", Eq. 7–9 | ⏳ Next |
| 4 | Self-supervised rapid-motion pretraining of the video backbone: pixel shuffling / random pixel replacement / block occlusion / Gaussian noise / temporal shuffling, weighted by landmark motion speed | Sec. 4, "Self-Supervised Pretraining for Rapid Motion Video Semantic Reconstruction", Eq. 3–6 | ⏳ Planned |
| 5 | Text Correction Network: a denoising seq2seq trained on artificially-corrupted sentences, applied to clean up Phase 2–4 output | Sec. 4, "Text Correction Network" | ⏳ Planned |
| 6 | Full benchmark run and comparison against the paper's reported numbers (BLEU-4 31.03 dev / 32.08 test on PHOENIX-2014-T) | Sec. 5, Table 2 | ⏳ Planned |

## Why start at Phase 2 instead of the full architecture?

Phases 3–5 each add real complexity (a second input modality, a custom
self-supervised training objective, a *second* trained network for
correction). They are much easier to design, debug, and ablate once a
simple video → text pipeline is already training end-to-end and producing
plausible sentences. Phase 2 is also a fair, well-defined baseline to
report progress against — every later phase should be measured as
"BLEU-4 improvement over the Phase 2 baseline," which is a clean story to
present.

## Design decisions worth flagging to your advisor

- **2D vs 3D backbone.** The paper uses a 3D ResNet18 so the CNN itself
  captures short-range motion. This baseline uses a 2D ResNet18 applied
  per-frame (cheaper to train on a single Colab GPU). Swapping in
  `torchvision.models.video.r3d_18` is a natural Phase 3/4 upgrade once
  the simpler pipeline is verified to work.
- **Tokenization.** The baseline uses word-level tokenization for
  simplicity. The paper doesn't specify subword tokenization either, but
  it's worth trying (e.g. SentencePiece) if the vocabulary of rare German
  compound words hurts BLEU.
- **Evaluation protocol.** The paper explicitly moves away from
  back-translation-only evaluation for the *production* (SLP) task
  because it hides pose errors (Sec. 1). That critique is specific to
  SLP; for the *translation* (SLT) task implemented here, BLEU/ROUGE/WER
  against ground-truth German text (as in Table 2) is the standard and
  correct protocol.
- **What's genuinely hard to reproduce exactly**: CNSign (the paper's own
  Chinese SLT dataset) is not yet publicly released (see the paper's
  footnote: "Access the download link from our GitHub repository upon
  release"), so this implementation targets PHOENIX-2014-T, the public
  benchmark the paper also reports on in Table 2. This is worth stating
  explicitly in your write-up.

## Suggested order of work from here

1. Get Phase 2 training and producing non-garbage sentences on a small
   subset (e.g. 200 training videos) before scaling to the full dataset —
   this catches data pipeline bugs fast, without long training times.
2. Once Phase 2's numbers are stable, add landmark extraction (MediaPipe)
   as a preprocessing step and implement Phase 3's fusion module.
3. Only after Phase 3 works, attempt Phase 4 (self-supervised
   pretraining) — it's the most novel and most fiddly part of the paper.
4. Phase 5 (text correction) is the most self-contained piece and could
   even be developed in parallel, as it's just a text-to-text denoising
   model trained on synthetic noisy/clean sentence pairs.
