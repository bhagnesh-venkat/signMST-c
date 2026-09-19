# Sign-to-Text Translation (SignMST-C baseline)

A from-scratch re-implementation of the **sign-to-text translation (SLT)**
half of:

> Li, Y., Zhang, Y., Tang, F., et al. *"Beyond Words: AuralLLM and
> SignMST-C for Precise Sign Language Production and Bidirectional
> Accessibility."* arXiv:2501.00765, 2025.

This repo currently implements **Phase 2** of the roadmap below: a
video → text Transformer baseline trained and evaluated on the public
**RWTH-PHOENIX-2014-T** benchmark, the same dataset the paper reports
results on (Table 2). See [`docs/ROADMAP.md`](docs/ROADMAP.md) for how
this maps onto the paper's full SignMST-C architecture and what's planned
next (multimodal landmark fusion, self-supervised pretraining, text
correction network).

## Project structure

```
sign-to-text-slt/
├── configs/config.yaml      # all paths & hyperparameters live here
├── docs/ROADMAP.md          # phase-by-phase plan mapped to the paper
├── src/
│   ├── vocab.py             # tokenizer + vocabulary
│   ├── dataset.py           # PHOENIX-2014-T PyTorch Dataset
│   ├── model.py             # CNN + Transformer encoder-decoder
│   ├── train.py             # training loop
│   ├── evaluate.py          # BLEU / ROUGE-L / WER evaluation
│   ├── inspect_data.py      # run this BEFORE training, sanity-checks paths
│   └── utils.py
├── notebooks/colab_train.ipynb  # ready-to-run Colab notebook
└── requirements.txt
```

## 1. Local setup (VSCode) — for editing code

You'll write/edit code locally, but train on Colab (see below) since
video models need a GPU. Locally you just need Python to run linting,
small tests, and `git`.

```bash
# clone your own repo after you've pushed it (see the Git section below)
git clone https://github.com/<your-username>/sign-to-text-slt.git
cd sign-to-text-slt

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In VSCode: install the **Python** extension, then `Ctrl+Shift+P` →
"Python: Select Interpreter" → choose `.venv`. That gives you linting,
autocomplete, and debugging against the exact same environment.

## 2. Quick smoke test (do this BEFORE you have real data)

Real access to PHOENIX-2014-T can take days/weeks to arrive. Don't wait on
it to prove your code works. This repo includes a script that generates a
tiny **fake** dataset (random noise images, made-up sentences) with the
exact same folder structure as the real thing:

```bash
python -m src.make_dummy_data
python -m src.inspect_data --config configs/config.yaml
python -m src.train --config configs/config.yaml
```

This trains for real (just on nonsense data), proves every part of the
pipeline runs end-to-end, and is exactly what you want to be able to
demo/screenshot for a progress check before the real dataset arrives.

## 3. Getting the real dataset

RWTH-PHOENIX-2014-T is a research dataset with its own license — you
need to request access from RWTH Aachen (it isn't a plain download link).
Start here: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX

Extract it so it matches `configs/config.yaml`'s `data.root`
(`data/phoenix2014T/` by default), then **run the sanity check before
anything else**:

```bash
python -m src.inspect_data --config configs/config.yaml
```

This prints the actual CSV column names and confirms the frame folders
resolve correctly. Dataset releases occasionally differ slightly in
folder naming — fix `configs/config.yaml` until this script's output
looks right, rather than guessing.

> `data/` is git-ignored on purpose — the dataset is large and licensed,
> so it should never be committed to your repo. On Colab you'll download
> it directly into the Colab VM instead (see the notebook).

## 4. Training on Google Colab (real data, real GPU)

Open `notebooks/colab_train.ipynb` in Colab (Runtime → Change runtime
type → GPU), which:
1. Clones your GitHub repo
2. Installs `requirements.txt`
3. Mounts Google Drive so you can persist the dataset + checkpoints
   between sessions (Colab's local disk is wiped when the runtime recycles)
4. Runs `src.inspect_data`, then `src.train`

Free Colab sessions time out after a few hours — `train.py` saves a
checkpoint every epoch (`train.save_every` in the config) specifically so
you can resume rather than losing progress.

## 5. Evaluating

```bash
python -m src.evaluate --config configs/config.yaml \
    --checkpoint checkpoints/epoch30.pt --split test
```

Prints BLEU-4 / ROUGE-L / WER and saves every prediction (with the
reference) to `checkpoints/predictions_test.json` so you can inspect
specific failure cases — useful evidence for a progress meeting.

## 6. Pushing this to GitHub from VSCode

1. Create a **new, empty** repository on github.com (no README/license —
   you already have files here). Copy its URL.
2. In VSCode, open this folder, then open the **Source Control** panel
   (`Ctrl+Shift+G`).
3. Click **Initialize Repository**.
4. Stage all files (the `+` icon), write a commit message like
   `"Phase 2: baseline SLT model"`, and commit.
5. Click **Publish Branch** (VSCode will ask for the GitHub URL/account if
   it's your first push) — or from the terminal:
   ```bash
   git remote add origin https://github.com/<your-username>/sign-to-text-slt.git
   git branch -M main
   git push -u origin main
   ```

From then on: edit → Source Control panel → stage → commit → push (the
cloud icon). Commit at the end of each roadmap phase so your history
itself documents progress — genuinely useful when explaining your work
to a supervisor.

## Metrics reference (paper, Table 2, PHOENIX-2014-T dev/test)

| Model | BLEU-4 |
|---|---|
| Joint-SLRT | 22.38 / 21.32 |
| STMC-T | 24.09 / 23.65 |
| SignBT | 24.45 / 24.32 |
| MMTLB | 27.61 / 28.39 |
| TwoStream-SLT | 28.66 / 28.95 |
| **SignMST-C (paper's full model)** | **31.03 / 32.08** |

This repo's Phase 2 baseline won't hit those numbers yet — it's missing
landmarks, the self-supervised pretraining, and the correction network
(Phases 3–5). Track your own BLEU-4 as you add each phase; that delta is
the actual research contribution to report.
