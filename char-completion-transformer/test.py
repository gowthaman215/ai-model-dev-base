"""Evaluate a trained checkpoint four ways:

  1. perplexity  -- on held-out text the model never saw
  2. grammar     -- do generated sentences obey the corpus grammar?
  3. novelty     -- is it composing, or regurgitating training lines?
  4. sweep       -- how sampling temperature trades coherence for variety

    python3 test.py                       # all four
    python3 test.py --only grammar        # one section
    python3 test.py --prompt "The river"  # interactive-ish continuation
"""

import argparse
import math
import os
import random
import re

import torch

import make_data
from model import GPT, GPTConfig
from train import CharTokenizer


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.chars = ckpt["chars"]
    tok.stoi = {c: i for i, c in enumerate(tok.chars)}
    tok.itos = {i: c for c, i in tok.stoi.items()}
    return model, cfg, tok


@torch.no_grad()
def perplexity(model, cfg, ids, device, stride=None):
    """Average next-token cross-entropy over a text, in non-overlapping windows."""
    stride = stride or cfg.block_size
    losses, counts = 0.0, 0
    for i in range(0, len(ids) - cfg.block_size - 1, stride):
        x = ids[i:i + cfg.block_size].unsqueeze(0).to(device)
        y = ids[i + 1:i + 1 + cfg.block_size].unsqueeze(0).to(device)
        _, loss = model(x, y)
        losses += loss.item() * cfg.block_size
        counts += cfg.block_size
    nll = losses / counts
    return nll, math.exp(nll)


@torch.no_grad()
def sample_text(model, tok, prompt, n_chars, device, temperature=0.8, top_k=None):
    idx = torch.tensor([tok.encode(prompt) or [0]], dtype=torch.long, device=device)
    out = model.generate(idx, n_chars, temperature, top_k)
    return tok.decode(out[0].tolist())


# a sentence is valid iff it matches the grammar that produced the corpus
def grammar_regex():
    def alt(words):
        return "(?:" + "|".join(re.escape(w) for w in words) + ")"
    subj_lower = alt(make_data.SUBJECTS)
    subj_upper = alt([s[0].upper() + s[1:] for s in make_data.SUBJECTS])
    core = f"{alt(make_data.VERBS)} {alt(make_data.OBJECTS)}(?: {alt(make_data.PLACES)})?"
    tail = f"(?:, {alt(make_data.CONJ)} {subj_lower} {core})?"
    return re.compile(f"^{subj_upper} {core}{tail}[.!?]$")


def split_sentences(text):
    """Drop the first and last fragments -- they are cut off by the sampling window."""
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines[1:-1] if ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--only", choices=["perplexity", "grammar", "novelty", "sweep"])
    ap.add_argument("--prompt", default=None, help="just continue this prompt and exit")
    ap.add_argument("--n-chars", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no checkpoint at {args.ckpt} -- run `python3 train.py` first")
    model, cfg, tok = load(args.ckpt, device)

    if args.prompt is not None:
        print(sample_text(model, tok, args.prompt, args.n_chars, device,
                          args.temperature, args.top_k))
        return

    run = lambda name: args.only in (None, name)

    # ---------------------------------------------------------- 1. perplexity
    if run("perplexity"):
        print("=" * 62)
        print("1. PERPLEXITY  (lower is better; 34 = random guessing)")
        print("=" * 62)
        text = open(args.data, encoding="utf-8").read()
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        n = int(0.9 * len(ids))

        # truly unseen text: same grammar, different random seed
        rng = random.Random(99)
        held = "\n".join(make_data.sentence(rng) for _ in range(2000)) + "\n"
        held_ids = torch.tensor(tok.encode(held), dtype=torch.long)

        for name, chunk in [("train split", ids[:n]),
                            ("val split (held out during training)", ids[n:]),
                            ("fresh corpus, unseen seed", held_ids)]:
            nll, ppl = perplexity(model, cfg, chunk, device)
            print(f"  {name:38s} loss {nll:.4f}   perplexity {ppl:6.3f}")
        print("  perplexity ~= how many characters the model is effectively torn between")

    # ------------------------------------------------------------- 2. grammar
    if run("grammar"):
        print("\n" + "=" * 62)
        print("2. GRAMMATICALITY  (does generated text obey the corpus grammar?)")
        print("=" * 62)
        pat = grammar_regex()
        text = sample_text(model, tok, "The", args.n_chars, device,
                           args.temperature, args.top_k)
        sents = split_sentences(text)
        bad = [s for s in sents if not pat.match(s)]
        ok = len(sents) - len(bad)
        print(f"  sampled {len(sents)} complete sentences at "
              f"temperature={args.temperature}, top_k={args.top_k}")
        print(f"  grammatical: {ok}/{len(sents)}  ({100 * ok / max(1, len(sents)):.1f}%)")
        for s in bad[:5]:
            print(f"    FAIL: {s}")
        print("\n  first 3 samples:")
        for s in sents[:3]:
            print(f"    {s}")

    # ------------------------------------------------------------- 3. novelty
    if run("novelty"):
        print("\n" + "=" * 62)
        print("3. NOVELTY  (composing new sentences, or copying training lines?)")
        print("=" * 62)
        train_lines = set(open(args.data, encoding="utf-8").read().split("\n"))
        text = sample_text(model, tok, "The", args.n_chars, device,
                           args.temperature, args.top_k)
        sents = split_sentences(text)
        seen = [s for s in sents if s in train_lines]
        print(f"  {len(sents) - len(seen)}/{len(sents)} sentences are NOT verbatim "
              f"training lines ({100 * (len(sents) - len(seen)) / max(1, len(sents)):.1f}% novel)")
        print(f"  note: the grammar has ~10*10*10*(1+8)*... combinations, so overlap")
        print(f"  with 6,000 training lines is expected to be low either way")

    # --------------------------------------------------------------- 4. sweep
    if run("sweep"):
        print("\n" + "=" * 62)
        print("4. TEMPERATURE SWEEP  (coherence vs. variety)")
        print("=" * 62)
        pat = grammar_regex()
        print(f"  {'temp':>6} {'top_k':>6} {'grammatical':>12}   sample")
        for temp, topk in [(0.2, None), (0.5, None), (0.8, None), (0.8, 40),
                           (1.0, None), (1.5, None)]:
            text = sample_text(model, tok, "The", 1500, device, temp, topk)
            sents = split_sentences(text)
            ok = sum(1 for s in sents if pat.match(s))
            rate = f"{ok}/{len(sents)}"
            first = (sents[0][:52] + "...") if sents and len(sents[0]) > 52 else (sents[0] if sents else "")
            print(f"  {temp:>6.1f} {str(topk):>6} {rate:>12}   {first}")


if __name__ == "__main__":
    main()
