#!/usr/bin/env python3
"""Visualize the weights of a transformer checkpoint, layer by layer.

Reads a safetensors checkpoint one tensor at a time (never holds the whole
model in RAM) and writes a set of PNGs showing where the parameters live,
how their values are distributed, and how those statistics change with depth.

Run it through the container launcher rather than directly:
    weight-visualizer run                       # defaults to gpt2 (124M, ~500MB)
    weight-visualizer run --model Qwen/Qwen2.5-0.5B
    weight-visualizer run --model /models/<local-dir> --out ./plots

Downloads land in the container's HF_HOME (/ai-cache -> cache/huggingface),
so nothing is written outside this workspace.
"""
import argparse
import os
import re
import sys
from collections import defaultdict


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

# Names differ per architecture (GPT-2 vs Llama vs Qwen), so classify on keywords.
ROLE_PATTERNS = [
    # `_emb` catches nanoGPT-style tok_emb / pos_emb; `ln_?\d` catches both
    # GPT-2's ln_1 and nanoGPT's ln1.
    ("embedding", r"(embed|_emb\b|wte|wpe|tok_emb|pos_emb)"),
    ("output_head", r"(lm_head|output\.weight)"),
    ("norm", r"(norm|ln_?\d|ln_f|layernorm)"),
    ("attention", r"(attn|attention)"),
    ("ffn", r"(mlp|ffn|feed_forward)"),
]
# Anchor with (^|\.) — GPT-2 names begin at "h.0.", with no leading dot.
LAYER_RE = re.compile(r"(?:^|\.)(?:h|layers|layer|blocks|block)\.(\d+)\.")

# Registered buffers, not learned parameters. GPT-2 stores its causal mask as a
# float32 tensor named `h.N.attn.bias` (1,1,ctx,ctx) — counting it inflates the
# attention bucket and the total (gpt2 reads as 137M instead of its real 124M).
# Note `attn.c_attn.bias` IS a real parameter, so these patterns anchor tightly.
BUFFER_RE = re.compile(
    r"(\.attn\.bias$|\.attn\.masked_bias$|masked_bias$|inv_freq$|\.rotary_emb\.)"
)


def classify(name):
    for role, pat in ROLE_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return role
    return "other"


def layer_index(name):
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def resolve_checkpoint(model):
    """Return a list of local .safetensors paths for a HF repo id or local dir."""
    if os.path.isdir(model):
        files = sorted(f for f in os.listdir(model) if f.endswith(".safetensors"))
        if not files:
            sys.exit(f"no .safetensors found in {model}")
        return [os.path.join(model, f) for f in files]

    from huggingface_hub import list_repo_files, hf_hub_download
    shards = [f for f in list_repo_files(model) if f.endswith(".safetensors")]
    if not shards:
        sys.exit(f"no .safetensors in repo {model} (older repos may be .bin only)")
    print(f"downloading {len(shards)} shard(s) from {model} ...")
    return [hf_hub_download(model, f) for f in sorted(shards)]


def collect(ckpt, sample_cap=200_000):
    """One pass over every tensor. Keeps only summary stats plus a small sample
    of values per tensor, so peak memory stays at roughly one tensor."""
    stats, samples, matrices = [], {}, {}
    skipped = []
    rng = np.random.default_rng(0)

    if True:
        if True:
            for name in ckpt.keys():
                if BUFFER_RE.search(name):
                    skipped.append(name)
                    continue
                t = ckpt.tensor(name)
                flat = t.ravel()

                # Subsample large tensors; histograms don't need 50M points.
                if flat.size > sample_cap:
                    idx = rng.choice(flat.size, sample_cap, replace=False)
                    sample = flat[idx]
                else:
                    sample = flat

                role = classify(name)
                stats.append({
                    "name": name, "role": role, "layer": layer_index(name),
                    "shape": tuple(t.shape), "count": int(flat.size),
                    "mean": float(flat.mean()), "std": float(flat.std()),
                    "absmax": float(np.abs(flat).max()),
                })
                samples[name] = sample.copy()

                # Keep one representative 2D matrix per role for the heatmaps.
                if t.ndim == 2 and role not in matrices:
                    matrices[role] = (name, t[:128, :128].copy())

                del t, flat, sample
    return stats, samples, matrices, skipped


def plot_budget(stats, out):
    by_role = defaultdict(int)
    for s in stats:
        by_role[s["role"]] += s["count"]
    total = sum(by_role.values())
    roles = sorted(by_role, key=by_role.get, reverse=True)
    vals = [by_role[r] / 1e6 for r in roles]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(roles, vals, color="#4C72B0")
    for b, r in zip(bars, roles):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{100 * by_role[r] / total:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("parameters (millions)")
    ax.set_title(f"Where the parameters live  —  {total / 1e6:.1f}M total")
    fig.tight_layout()
    fig.savefig(f"{out}/01_parameter_budget.png", dpi=130)
    plt.close(fig)


def plot_distributions(stats, samples, out):
    """One histogram per role, on a log y-axis so the tails are visible."""
    roles, seen = [], set()
    for s in stats:
        if s["role"] not in seen and s["count"] > 1000:
            seen.add(s["role"])
            roles.append(s)

    n = len(roles)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), roles):
        ax.hist(samples[s["name"]], bins=120, color="#55A868", log=True)
        ax.set_title(f"{s['role']}\n{s['name'][-42:]}", fontsize=8)
        ax.set_xlabel("weight value")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Weight value distributions (log counts) — near-Gaussian, heavy tails", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{out}/02_distributions.png", dpi=130)
    plt.close(fig)


def plot_depth_trend(stats, out):
    """How weight statistics evolve from the first layer to the last."""
    series = defaultdict(lambda: defaultdict(list))
    for s in stats:
        if s["layer"] is not None and s["count"] > 1000:
            series[s["role"]][s["layer"]].append(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    plotted = 0
    for role, per_layer in sorted(series.items()):
        layers = sorted(per_layer)
        if len(layers) < 2:
            continue
        ax1.plot(layers, [np.mean([x["std"] for x in per_layer[l]]) for l in layers],
                 marker="o", ms=3, label=role)
        ax2.plot(layers, [max(x["absmax"] for x in per_layer[l]) for l in layers],
                 marker="o", ms=3, label=role)
        plotted += 1
    ax1.set_xlabel("layer"); ax1.set_ylabel("std of weights"); ax1.set_title("Spread by depth")
    ax2.set_xlabel("layer"); ax2.set_ylabel("max |weight|"); ax2.set_title("Outlier magnitude by depth")
    for ax in (ax1, ax2):
        if plotted:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/03_depth_trend.png", dpi=130)
    plt.close(fig)


def plot_heatmaps(matrices, out):
    """The 'it looks like static' demonstration — meaning is not in single entries."""
    items = sorted(matrices.items())
    n = len(items)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
    for ax, (role, (name, block)) in zip(axes.ravel(), items):
        lim = np.abs(block).max()
        im = ax.imshow(block, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{role}  (128x128 crop)\n{name[-38:]}", fontsize=8)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Raw weight matrices — structureless to the eye, by design", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{out}/04_heatmaps.png", dpi=130)
    plt.close(fig)


def plot_outlier_channels(stats, samples, out):
    """Per-output-channel max|w|. The spikes here are what break naive quantization."""
    cands = [s for s in stats if len(s["shape"]) == 2 and s["role"] in ("ffn", "attention")]
    if not cands:
        return
    s = max(cands, key=lambda x: x["count"])
    vals = samples[s["name"]]
    # samples may be subsampled, so recompute a per-column proxy from the sample
    k = min(2048, vals.size)
    per_chan = np.abs(vals[:k].reshape(-1, min(64, k)))
    chan_max = per_chan.max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(len(chan_max)), chan_max, color="#C44E52", width=1.0)
    ax.axhline(np.median(chan_max), ls="--", c="k", lw=1,
               label=f"median {np.median(chan_max):.3f}")
    ax.set_xlabel("channel (sampled)")
    ax.set_ylabel("max |weight|")
    ax.set_title(f"Outlier channels — {s['name'][-46:]}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/05_outlier_channels.png", dpi=130)
    plt.close(fig)


def print_summary(stats, skipped):
    total = sum(s["count"] for s in stats)
    print(f"\n{'tensor':<52}{'shape':<20}{'params':>12}{'std':>9}{'absmax':>9}")
    print("-" * 102)
    for s in sorted(stats, key=lambda x: -x["count"])[:15]:
        print(f"{s['name'][-50:]:<52}{str(s['shape']):<20}{s['count']:>12,}"
              f"{s['std']:>9.4f}{s['absmax']:>9.3f}")
    print("-" * 102)
    print(f"{'TOTAL':<52}{'':<20}{total:>12,}   ({total / 1e6:.1f}M parameters, "
          f"{len(stats)} tensors)")
    if skipped:
        print(f"skipped {len(skipped)} non-parameter buffer(s), e.g. {skipped[0]}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace repo id or local directory (default: gpt2)")
    p.add_argument("--out", default=None,
                   help="output directory for PNGs (default: ./plots/<model>)")
    p.add_argument("--view-mode", choices=("plots", "anatomy", "all"), default="all",
                   help="plots = the five PNG plates; anatomy = the scalable "
                        "vector anatomy map; all = both (default)")
    p.add_argument("--json", metavar="PATH", default=None,
                   help="also write per-tensor geometry as JSON")
    p.add_argument("--px-scale", type=float, default=None,
                   help="pixels per matrix element in the anatomy map; "
                        "auto-fits when unset")
    p.add_argument("--canvas-width", type=float, default=None,
                   help="fix the anatomy block-panel width so several models "
                        "render at a comparable scale")
    args = p.parse_args()

    if args.out is None:
        slug = os.path.splitext(os.path.basename(args.model.rstrip("/")))[0]
        args.out = os.path.join("./plots", slug)
    os.makedirs(args.out, exist_ok=True)

    from loader import Checkpoint
    ckpt = Checkpoint(args.model)
    print(f"reading tensors ({ckpt.kind}) ...")
    stats, samples, matrices, skipped = collect(ckpt)
    print_summary(stats, skipped)

    if args.json:
        import json
        payload = {
            "model": args.model,
            "total": sum(s["count"] for s in stats),
            "skipped_buffers": len(skipped),
            "tensors": [{k: s[k] for k in
                         ("name", "role", "layer", "shape", "count", "std", "absmax")}
                        for s in stats],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(payload, fh)
        print(f"wrote {args.json}")

    if args.view_mode in ("plots", "all"):
        print("plotting ...")
        plot_budget(stats, args.out)
        plot_distributions(stats, samples, args.out)
        plot_depth_trend(stats, args.out)
        plot_heatmaps(matrices, args.out)
        plot_outlier_channels(stats, samples, args.out)

    if args.view_mode in ("anatomy", "all"):
        print("rendering anatomy ...")
        import anatomy
        layers = {s["layer"] for s in stats if s["layer"] is not None}
        meta = {"name": args.model, "variant": "",
                "layers": len(layers) or 1,
                "total": sum(s["count"] for s in stats)}
        svg = anatomy.render(stats, meta, canvas_w=args.canvas_width,
                             px=args.px_scale)
        with open(os.path.join(args.out, "anatomy.svg"), "w") as fh:
            fh.write(svg)

    for f in sorted(os.listdir(args.out)):
        print(f"  {os.path.join(args.out, f)}")


if __name__ == "__main__":
    main()
