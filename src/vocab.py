"""
Simple whitespace tokenizer + vocabulary for the SLT target language (German).

This is intentionally simple (word-level, not subword) so a beginner can read
every line of it. Swapping in a subword tokenizer (e.g. SentencePiece) is a
natural later improvement -- see docs/ROADMAP.md.
"""
import json
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def tokenize(text: str, lowercase: bool = True):
    if lowercase:
        text = text.lower()
    return text.strip().split()


class Vocab:
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi or {}
        self.itos = itos or []

    @classmethod
    def build(cls, sentences, min_freq: int = 2, lowercase: bool = True):
        counter = Counter()
        for s in sentences:
            counter.update(tokenize(s, lowercase))

        itos = list(SPECIAL_TOKENS)
        for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq:
                itos.append(tok)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi, itos)

    def encode(self, text: str, lowercase: bool = True, add_special: bool = True):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokenize(text, lowercase)]
        if add_special:
            ids = [self.stoi[BOS]] + ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids, strip_special: bool = True):
        toks = []
        for i in ids:
            tok = self.itos[i] if i < len(self.itos) else UNK
            if strip_special and tok in SPECIAL_TOKENS:
                if tok == EOS:
                    break
                continue
            toks.append(tok)
        return " ".join(toks)

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi, itos)
