#!/usr/bin/env python3
"""Check container dependencies and mounts without downloading a model."""

import importlib
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def main():
    errors = []
    if not Path("/.dockerenv").exists():
        errors.append("Run this check inside the AI Docker container.")
    print(f"Python:       {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        errors.append("Expected Python 3.11.")

    packages = (
        "torch", "transformers", "safetensors", "accelerate", "huggingface_hub",
        "tokenizers", "numpy", "matplotlib", "jupyterlab", "ipykernel", "debugpy",
    )
    for name in packages:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {module.__version__}")
            if name == "torch":
                # Exercise an actual CPU operation, not just the import.
                assert (module.ones(2, 2) @ module.ones(2, 2)).sum().item() == 8
                print(f"CUDA:         {module.cuda.is_available()}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    models = Path(os.environ.get("MODEL_DIR", "/models"))
    cache = Path(os.environ.get("HF_HOME", "/ai-cache"))
    for path in (models, cache):
        if not path.is_dir() or not os.path.ismount(path):
            errors.append(f"Missing bind mount: {path}")
    if models.is_dir():
        if not os.statvfs(models).f_flag & os.ST_RDONLY:
            errors.append(f"Models mount must be read-only: {models}")
        else:
            print(f"Models:       {models} (host-mounted, read-only)")
    try:
        with tempfile.TemporaryFile(dir=cache) as probe:
            probe.write(b"AI cache verification\n")
            probe.flush()
        print(f"HF cache:     {cache} (host-mounted, writable)")
    except OSError as exc:
        errors.append(f"HF cache is not writable: {exc}")

    root = Path(os.environ.get("AI_MODEL_DEV_ROOT", "/workspace/ai-model-dev"))
    if not root.is_dir() or not os.path.ismount(root):
        errors.append(f"Missing repository bind mount: {root}")
    for command in ("char-completion-transformer", "weight-visualizer", "jupyter"):
        location = shutil.which(command)
        if location:
            print(f"Command:      {command} -> {location}")
        else:
            errors.append(f"Command missing from PATH: {command}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("AI container is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
