"""Train with fixed settings on every UTF-8 text file under data/."""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
import sys


DATA_DIR = Path("data")
OUT_DIR = Path("out")
ARCHIVE_DIR = OUT_DIR / "archive"
RUN_ARTIFACTS = ("ckpt.pt", "combined_corpus.txt", "training_files.txt")


def training_command(run_dir, combined_corpus):
    """Build the fixed command without exposing internal tuning parameters."""
    return [
        sys.executable, "train.py",
        "--data", str(combined_corpus),
        "--out-dir", str(run_dir),
        "--steps", "2000",
        "--batch-size", "32",
        "--block-size", "128",
        "--n-layer", "4",
        "--n-head", "4",
        "--n-embd", "128",
        "--dropout", "0.1",
        "--lr", "0.001",
        "--eval-every", "200",
        "--prompt", "The",
        "--max-new-tokens", "300",
        "--temperature", "0.8",
        "--top-k", "40",
    ]


def read_text_file(path):
    """Return UTF-8 text, or None for binary/non-UTF-8/empty files."""
    try:
        raw = path.read_bytes()
        if not raw or b"\x00" in raw:
            return None
        return raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def archive_existing_run():
    existing = [OUT_DIR / name for name in RUN_ARTIFACTS if (OUT_DIR / name).exists()]
    if not existing:
        print("No existing training artifacts to archive.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archived_run = ARCHIVE_DIR / timestamp
    archived_run.mkdir(parents=True, exist_ok=False)
    for source in existing:
        source.replace(archived_run / source.name)
    print(f"Preserved previous training bundle as: {archived_run}")
    for name in RUN_ARTIFACTS:
        archived = archived_run / name
        if archived.exists():
            print(f"  + {archived}")
    return archived_run


def collect_corpus():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Training directory does not exist: {DATA_DIR}")

    included = []
    skipped = []
    chunks = []

    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        text = read_text_file(path)
        if text is None:
            skipped.append(path)
            continue
        included.append(path)
        chunks.append(text.rstrip() + "\n")

    if not included:
        raise SystemExit(f"No non-empty UTF-8 text files found under {DATA_DIR}")

    corpus = "\n".join(chunks)
    if len(corpus) < 1_500:
        raise SystemExit(
            f"Combined data has only {len(corpus):,} characters. "
            "Add at least 1,500 characters so training and validation batches fit."
        )

    manifest_lines = ["sha256\tbytes\tpath"]
    for path in included:
        raw = path.read_bytes()
        manifest_lines.append(
            f"{hashlib.sha256(raw).hexdigest()}\t{len(raw)}\t{path}"
        )
    manifest = "\n".join(manifest_lines) + "\n"

    print(f"Included {len(included)} training file(s):")
    for path in included:
        print(f"  + {path}")
    for path in skipped:
        print(f"  - skipped binary, non-UTF-8, or empty file: {path}")
    print(f"Combined corpus size: {len(corpus):,} characters")
    return corpus, manifest


def training_data_unchanged(corpus, manifest):
    checkpoint = OUT_DIR / "ckpt.pt"
    combined_corpus = OUT_DIR / "combined_corpus.txt"
    training_manifest = OUT_DIR / "training_files.txt"
    if not all(path.is_file() for path in (checkpoint, combined_corpus, training_manifest)):
        return False
    return (
        combined_corpus.read_text(encoding="utf-8") == corpus
        and training_manifest.read_text(encoding="utf-8") == manifest
    )


def write_training_records(run_dir, corpus, manifest):
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_corpus = run_dir / "combined_corpus.txt"
    training_manifest = run_dir / "training_files.txt"
    combined_corpus.write_text(corpus, encoding="utf-8")
    training_manifest.write_text(manifest, encoding="utf-8")
    print(f"Combined corpus: {combined_corpus}")
    print(f"File manifest:   {training_manifest}")
    return combined_corpus


def main():
    parser = argparse.ArgumentParser(
        description="Train all UTF-8 text files under data/ using preset settings."
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="archive the old checkpoint and its training-data records before training",
    )
    args = parser.parse_args()

    run_dir = OUT_DIR
    checkpoint = OUT_DIR / "ckpt.pt"
    corpus, manifest = collect_corpus()
    if training_data_unchanged(corpus, manifest):
        print("Training data is unchanged; keeping the existing model and doing nothing.")
        return

    if not args.no_overwrite and checkpoint.exists():
        print(f"Warning: this run can replace the existing {checkpoint}")
    if args.no_overwrite:
        archive_existing_run()
    combined_corpus = write_training_records(run_dir, corpus, manifest)
    print("Starting training with the project's preset configuration...\n")
    subprocess.run(training_command(run_dir, combined_corpus), check=True)
    print(f"\nTraining complete. Model checkpoint: {checkpoint}")
    print("Run `python prompt.py` to try it interactively.")


if __name__ == "__main__":
    main()
