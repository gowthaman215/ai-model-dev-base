"""Generate a small, self-contained training corpus (no download needed).

Sentences are sampled from a tiny grammar, so a character-level model has real
structure to learn: spelling, word boundaries, punctuation and word order.
Swap this out for any plain-text file of your own via `train.py --data`.
"""

import argparse
import random

SUBJECTS = ["the cat", "a robot", "my neighbour", "the old sailor", "every student",
            "the tired engineer", "a small bird", "the river", "her brother", "the machine"]
VERBS = ["watched", "repaired", "remembered", "carried", "questioned",
         "followed", "painted", "counted", "abandoned", "measured"]
OBJECTS = ["the broken clock", "three yellow boats", "a quiet street", "the morning light",
           "an empty notebook", "the long bridge", "seventeen letters", "a rusty key",
           "the sleeping town", "her favourite song"]
PLACES = ["by the harbour", "under the bridge", "in the winter", "before sunrise",
          "near the old mill", "after the storm", "on a tuesday", "without a word"]
CONJ = ["and", "but", "so", "while", "because"]


def sentence(rng):
    s = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}"
    if rng.random() < 0.6:
        s += f" {rng.choice(PLACES)}"
    if rng.random() < 0.35:
        s += (f", {rng.choice(CONJ)} {rng.choice(SUBJECTS)} "
              f"{rng.choice(VERBS)} {rng.choice(OBJECTS)}")
    return s[0].upper() + s[1:] + rng.choice([".", ".", ".", "!", "?"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--sentences", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lines = [sentence(rng) for _ in range(args.sentences)]

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {args.sentences} sentences, "
          f"{sum(len(l) + 1 for l in lines):,} characters")


if __name__ == "__main__":
    main()
