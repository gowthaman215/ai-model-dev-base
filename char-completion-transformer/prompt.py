"""Load a trained checkpoint once and accept prompts repeatedly."""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from train import CharTokenizer


CHECKPOINT = Path("out/ckpt.pt")
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 40


def resolve_checkpoint(requested):
    if requested:
        return Path(requested)
    return CHECKPOINT


def load_model(checkpoint_path):
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"No trained model found at {checkpoint_path}. Run `python train_all.py` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.chars = checkpoint["chars"]
    tokenizer.stoi = {char: index for index, char in enumerate(tokenizer.chars)}
    tokenizer.itos = {index: char for char, index in tokenizer.stoi.items()}
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Prompt a trained model repeatedly.")
    parser.add_argument(
        "--checkpoint",
        help="checkpoint to load; defaults to out/ckpt.pt",
    )
    args = parser.parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model, tokenizer, device = load_model(checkpoint_path)
    parameter_count = model.num_params()
    print(f"Loaded {checkpoint_path} ({parameter_count:,} parameters) on {device}.")
    print("Enter a text beginning for the model to continue.")
    print("Commands: /help, /quit")

    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break
        if prompt.lower() == "/help":
            print("Type a short prefix such as: The river")
            print("This is a completion model, not an instruction-following chatbot.")
            continue

        unknown = sorted(set(prompt) - set(tokenizer.chars))
        if unknown:
            print(f"Note: characters not present in training data are ignored: {unknown!r}")

        encoded = tokenizer.encode(prompt)
        if not encoded:
            print("The prompt has no characters known by this model. Try another prompt.")
            continue

        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )
        print("\ncompletion>")
        print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
