"""Train the character-level GPT in model.py and sample from it.

    python train.py                       # train on the bundled synthetic corpus
    python train.py --data mytext.txt     # train on any plain-text file
    python train.py --sample-only         # load out/ckpt.pt and just generate
    python train.py --track               # also record how the weights move
"""

import argparse
import math
import os
import time

import torch

from model import GPT, GPTConfig


class CharTokenizer:
    """The simplest tokenizer that works: one token per character."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def get_batch(data, block_size, batch_size, device):
    """Sample random windows; targets are the inputs shifted one step right."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, batch_size, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, total, base_lr, warmup):
    """Linear warmup, then cosine decay to 10% of the base learning rate."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--track", action="store_true",
                    help="record per-tensor weight movement and plot the trajectory")
    ap.add_argument("--track-every", type=int, default=0,
                    help="tracking interval in steps (default: same as --eval-every)")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    # ---------------------------------------------------------------- sample only
    if args.sample_only:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        tok = CharTokenizer.__new__(CharTokenizer)
        tok.chars = ckpt["chars"]
        tok.stoi = {c: i for i, c in enumerate(tok.chars)}
        tok.itos = {i: c for c, i in tok.stoi.items()}
        prompt = torch.tensor([tok.encode(args.prompt) or [0]], dtype=torch.long, device=device)
        out = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(tok.decode(out[0].tolist()))
        return

    # -------------------------------------------------------------------- data
    if not os.path.exists(args.data):
        raise SystemExit(
            f"no corpus at {args.data} -- run `python make_data.py` first, "
            f"or pass --data <your-text-file>"
        )
    text = open(args.data, encoding="utf-8").read()
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    splits = {"train": ids[:n], "val": ids[n:]}
    print(f"corpus: {len(text):,} chars | vocab: {tok.vocab_size} | "
          f"train/val tokens: {len(splits['train']):,}/{len(splits['val']):,}")

    # ------------------------------------------------------------------- model
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters on {device}")

    # no weight decay on 1-D params (biases, LayerNorm gains) -- standard GPT recipe
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    # ------------------------------------------------------------- training loop
    os.makedirs(args.out_dir, exist_ok=True)

    tracker = None
    if args.track:
        from tracker import Tracker
        tracker = Tracker(model, args.out_dir,
                          every=args.track_every or args.eval_every)
        print(f"tracking {len(tracker.names)} weight tensors "
              f"every {tracker.every} steps")

    best_val = float("inf")
    warmup = max(1, args.steps // 20)
    t0 = time.time()

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warmup)

        x, y = get_batch(splits["train"], cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ returns the total norm *before* clipping — the honest
        # gradient magnitude, and free to record.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        ratios = {}
        measuring = tracker is not None and tracker.should_measure(step, args.steps)
        if measuring:
            tracker.snapshot(model)          # w before the step
        opt.step()
        if measuring:
            ratios = tracker.measure_update(model)   # against w after the step

        if tracker is not None:
            tracker.log_step(step, loss.item(),
                             opt.param_groups[0]["lr"], grad_norm)

        if step % args.eval_every == 0 or step == args.steps - 1:
            m = estimate_loss(model, splits, cfg, args.batch_size, device)
            if tracker is not None:
                tracker.log_eval(step, model, ratios, m["train"], m["val"], grad_norm)
            print(f"step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
            if m["val"] < best_val:
                best_val = m["val"]
                torch.save(
                    {"model": model.state_dict(), "config": vars(cfg), "chars": tok.chars},
                    ckpt_path,
                )

    print(f"\nbest val loss {best_val:.4f} -> {ckpt_path}")

    if tracker is not None:
        tracker.save()
        tracker.summary()

    # ---------------------------------------------------------------- sampling
    print(f"\n--- sample (prompt: {args.prompt!r}) ---")
    prompt = torch.tensor([tok.encode(args.prompt) or [0]], dtype=torch.long, device=device)
    out = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
