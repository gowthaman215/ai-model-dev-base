"""Record how the weights actually move during training.

Two sampling rates, because the two things cost very different amounts:

  every step      loss, learning rate, and the pre-clip gradient norm. All three
                  are already computed by the training loop, so this is free.
  every N steps   per-tensor weight norm, standard deviation, and the
                  update-to-weight ratio. This clones the tracked tensors once,
                  so it is cheap but not free.

The update-to-weight ratio is the headline diagnostic:

    ratio = || w_after - w_before || / || w_before ||    for one optimizer step

It should sit near 1e-3. Much above 1e-2 and the learning rate is too high —
expect loss spikes. Much below 1e-4 and that tensor has stopped learning, which
loss alone will not show you because the other layers cover for it.
"""
import json
import os

import torch


class Tracker:
    def __init__(self, model, out_dir, every=100):
        self.out_dir = out_dir
        self.every = every
        # 2-D weights only; biases and LayerNorm gains are too small to be
        # informative and would clutter every plot.
        self.names = [n for n, p in model.named_parameters() if p.dim() >= 2]
        self.steps = []          # per-step: step, loss, lr, grad_norm
        self.evals = []          # per-interval: per-tensor statistics
        self._before = None

    # ---------------------------------------------------------------- per step

    def should_measure(self, step, total):
        return step % self.every == 0 or step == total - 1

    def log_step(self, step, loss, lr, grad_norm):
        self.steps.append({
            "step": step,
            "loss": float(loss),
            "lr": float(lr),
            "grad_norm": float(grad_norm),
        })

    # ------------------------------------------------- around the optimizer step

    def snapshot(self, model):
        """Clone tracked weights before opt.step() so the update can be measured."""
        d = dict(model.named_parameters())
        self._before = {n: d[n].detach().clone() for n in self.names}

    def measure_update(self, model):
        """Called straight after opt.step(). Returns {name: ratio}."""
        if self._before is None:
            return {}
        d = dict(model.named_parameters())
        out = {}
        for n in self.names:
            w0 = self._before[n]
            w1 = d[n].detach()
            denom = w0.norm().item()
            out[n] = (w1 - w0).norm().item() / denom if denom > 0 else 0.0
        self._before = None
        return out

    # ---------------------------------------------------------------- per eval

    def log_eval(self, step, model, ratios, train_loss, val_loss, grad_norm):
        d = dict(model.named_parameters())
        tensors = {}
        for n in self.names:
            w = d[n].detach()
            g = d[n].grad
            tensors[n] = {
                "norm": w.norm().item(),
                "std": w.std().item(),
                "absmax": w.abs().max().item(),
                "ratio": ratios.get(n, float("nan")),
                "grad_norm": g.norm().item() if g is not None else float("nan"),
            }
        self.evals.append({
            "step": step, "train": train_loss, "val": val_loss,
            "gap": val_loss - train_loss, "grad_norm": float(grad_norm),
            "tensors": tensors,
        })

    # ------------------------------------------------------------------ output

    def save(self, plot=True):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, "track.json")
        with open(path, "w") as f:
            json.dump({"names": self.names, "every": self.every,
                       "steps": self.steps, "evals": self.evals}, f)
        print(f"tracking data -> {path}")
        if plot and self.evals:
            self._plot()
        return path

    @staticmethod
    def _role(name):
        if "attn" in name:
            return "attention"
        if "mlp" in name or "ffn" in name:
            return "feed-forward"
        if "emb" in name:
            return "embedding"
        return "other"

    def _plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        COLOR = {"attention": "#4C72B0", "feed-forward": "#DD8452",
                 "embedding": "#8172B3", "other": "#937860"}
        ev = self.evals
        xs = [e["step"] for e in ev]
        sx = [s["step"] for s in self.steps]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

        # --- loss and the generalization gap ---
        ax = axes[0][0]
        ax.plot(xs, [e["train"] for e in ev], marker="o", ms=3, label="train")
        ax.plot(xs, [e["val"] for e in ev], marker="o", ms=3, label="val")
        ax.set_title("Loss")
        ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (nats)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # A line, not a filled area: the gap is often ~1e-3 and filling to zero
        # turns that into alarming-looking spikes.
        g2 = ax.twinx()
        g2.plot(xs, [e["gap"] for e in ev], lw=1.2, ls=":", color="#C44E52",
                marker="s", ms=2.5)
        g2.axhline(0, lw=0.8, color="#C44E52", alpha=0.4)
        g2.set_ylabel("val - train (overfit gap)", fontsize=9, color="#C44E52")
        g2.tick_params(labelsize=8, colors="#C44E52")

        # --- update-to-weight ratio, the health metric ---
        ax = axes[0][1]
        seen = set()
        for n in self.names:
            role = self._role(n)
            ys = [e["tensors"][n]["ratio"] for e in ev]
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1, color=COLOR[role],
                    alpha=0.8, label=role if role not in seen else None)
            seen.add(role)
        ax.axhspan(1e-4, 1e-2, color="#55A868", alpha=0.1)
        ax.axhline(1e-3, ls="--", lw=1, c="#55A868")
        ax.set_yscale("log")
        ax.set_title("Update : weight ratio  (healthy band shaded, 1e-3 target)")
        ax.set_xlabel("step"); ax.set_ylabel("||Δw|| / ||w||")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # --- weight norm drift ---
        ax = axes[1][0]
        seen = set()
        for n in self.names:
            role = self._role(n)
            ys = [e["tensors"][n]["norm"] for e in ev]
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1, color=COLOR[role],
                    alpha=0.8, label=role if role not in seen else None)
            seen.add(role)
        ax.set_title("Weight norm per tensor")
        ax.set_xlabel("step"); ax.set_ylabel("||w||")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # --- gradient norm, every step ---
        ax = axes[1][1]
        ax.plot(sx, [s["grad_norm"] for s in self.steps], lw=0.7, color="#4C72B0")
        ax.axhline(1.0, ls="--", lw=1, c="#C44E52", label="clip threshold")
        ax.set_yscale("log")
        ax.set_title("Gradient norm before clipping (every step)")
        ax.set_xlabel("step"); ax.set_ylabel("||g||")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig.tight_layout()
        out = os.path.join(self.out_dir, "training_trajectory.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"trajectory plot  -> {out}")

    # ------------------------------------------------------------------ console

    def summary(self):
        if not self.evals:
            return
        last = self.evals[-1]
        ratios = [(n, t["ratio"]) for n, t in last["tensors"].items()]
        ratios.sort(key=lambda kv: kv[1])
        print("\n--- weight movement at the final measured step ---")
        print(f"{'tensor':<34}{'||w||':>10}{'std':>9}{'Δw/w':>11}  verdict")
        for n, r in ratios:
            t = last["tensors"][n]
            if r > 1e-2:
                v = "HIGH - lr may be too large"
            elif r < 1e-4:
                v = "LOW - barely learning"
            else:
                v = "healthy"
            print(f"{n[:32]:<34}{t['norm']:>10.3f}{t['std']:>9.4f}{r:>11.2e}  {v}")
        print(f"\noverfit gap (val - train): {last['gap']:+.4f}")
