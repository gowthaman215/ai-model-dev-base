#!/usr/bin/env python3
"""One accessor over both checkpoint formats.

safetensors is memory-mapped, so tensors and sub-rectangles are read lazily off
disk. A PyTorch .pt is a pickle — there is no way to read part of it, so the
whole state dict is loaded once and held in memory. That is fine for the small
checkpoints this tool targets; for a multi-GB .pt, convert to safetensors first
(`Checkpoint.to_safetensors`) and the lazy path applies again.
"""
import os

import numpy as np

PT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin")
# Keys a training script commonly nests the weights under.
SD_KEYS = ("model", "state_dict", "model_state_dict", "net", "weights")


class Checkpoint:
    def __init__(self, source):
        self.source = source
        self.kind = None
        self._paths = []
        self._mem = {}
        self._shapes = {}
        self._open(source)

    # ---------- loading ----------

    def _open(self, source):
        if os.path.isfile(source) and source.endswith(PT_SUFFIXES):
            self._open_pt(source)
        else:
            self._open_safetensors(source)

    def _open_pt(self, path):
        import torch
        self.kind = "pt"
        # weights_only=True is the safe default; fall back for checkpoints that
        # pickle a config dict or char list alongside the tensors.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)

        sd = obj
        if isinstance(obj, dict):
            for k in SD_KEYS:
                if isinstance(obj.get(k), dict):
                    sd = obj[k]
                    break
        if not isinstance(sd, dict):
            raise SystemExit(f"{path}: no state dict found (top level is {type(obj).__name__})")

        for name, t in sd.items():
            if not hasattr(t, "numpy"):
                continue
            arr = t.detach().to("cpu").float().numpy()
            self._mem[name] = arr
            self._shapes[name] = tuple(arr.shape)
        if not self._mem:
            raise SystemExit(f"{path}: state dict holds no tensors")

    def _open_safetensors(self, source):
        from visualize import resolve_checkpoint
        self.kind = "safetensors"
        self._paths = resolve_checkpoint(source)
        from safetensors import safe_open
        for p in self._paths:
            with safe_open(p, framework="np") as f:
                for name in f.keys():
                    self._shapes[name] = tuple(f.get_slice(name).get_shape())
                    self._mem[name] = p          # value is the shard path

    # ---------- access ----------

    def keys(self):
        return list(self._shapes)

    def shape(self, name):
        return self._shapes[name]

    def tensor(self, name):
        """The whole tensor as float32 ndarray."""
        if self.kind == "pt":
            return self._mem[name]
        from safetensors import safe_open
        with safe_open(self._mem[name], framework="np") as f:
            a = np.asarray(f.get_tensor(name))
        return a.astype(np.float32, copy=False)

    def window(self, name, r0, r1, c0, c1):
        """A 2D sub-rectangle, read lazily where the format allows it."""
        if self.kind == "pt":
            return self._mem[name][r0:r1, c0:c1]
        from safetensors import safe_open
        with safe_open(self._mem[name], framework="np") as f:
            a = np.asarray(f.get_slice(name)[r0:r1, c0:c1])
        return a.astype(np.float32, copy=False)

    # ---------- conversion ----------

    def to_safetensors(self, out_path):
        """Write the same tensors as safetensors, restoring the lazy path."""
        from safetensors.numpy import save_file
        save_file({k: np.ascontiguousarray(self.tensor(k)) for k in self.keys()},
                  out_path)
        return out_path
