# char-completion-transformer

A minimal, readable decoder-only Transformer (GPT-style) language model in PyTorch —
the same architecture every modern LLM uses, scaled down to ~0.8M parameters so it
trains on a laptop CPU in a few minutes.

For a full step-by-step explanation of how the model is built, trained, and
tested — with measured numbers — see **[WALKTHROUGH.md](WALKTHROUGH.md)**.

```
model.py       the Transformer itself (~150 lines)
train.py       char tokenizer + training loop + sampling
train_all.py   trains every UTF-8 text file under data/ with preset settings
prompt.py      repeatedly prompts one loaded checkpoint in the terminal
make_data.py   generates a small synthetic corpus (no download needed)
test.py        perplexity / grammaticality / novelty / temperature sweep
```

## Easy development workflow

Use this three-step cycle whenever you want to teach and test the model.

### 1. Add knowledge

Place one or more non-empty UTF-8 text files anywhere under `data/`:

```text
data/
├── corpus.txt
├── product_notes.txt
└── documentation/
    └── component_guide.md
```

`train_all.py` reads the directory recursively. Binary, empty, and non-UTF-8
files are skipped.

### 2. Train

```bash
python train_all.py --no-overwrite
```

The command combines the source text, trains using the project's preset
parameters, and writes the newest model to `out/ckpt.pt`.

- If the training data is unchanged, the command does nothing.
- If the data changed, the previous model bundle is preserved under
  `out/archive/<timestamp>/`.
- Every bundle keeps `ckpt.pt`, `combined_corpus.txt`, and
  `training_files.txt` together.
- `training_files.txt` records the source path, byte size, and SHA-256 hash of
  every included knowledge file.

Check exactly what trained the current model:

```bash
cat out/training_files.txt
```

To also record how the weights move during the run, add `--track` — see
[Tracking how the weights move](#tracking-how-the-weights-move).

### 3. Prompt and check the model

```bash
python prompt.py
```

Enter text beginnings repeatedly:

```text
prompt> The river
completion>
The river followed the morning light...

prompt> Every student
completion>
Every student repaired...

prompt> /quit
Goodbye.
```

This is a small character-level completion model. It learns spelling, sentence
structure, vocabulary, and patterns from the supplied text, but it is not a
ChatGPT-style factual question-answering or instruction-following model.

## Host launcher

Run the container launcher directly from the `ai-model-dev` directory:

```bash
cd ~/10718695/github/ai-model-dev
./ai-dev-container.sh build
./ai-dev-container.sh verify
./ai-dev-container.sh shell
./ai-dev-container.sh jupyter
```

Host commands manage the shared container. Project operations run after entering
the container. To use another Jupyter host port:

```bash
AI_JUPYTER_PORT=8889 ./ai-dev-container.sh jupyter
```

After entering the shared environment with `./ai-dev-container.sh shell`, use the
project-specific command:

```bash
char-completion-transformer build
char-completion-transformer train
char-completion-transformer prompt
char-completion-transformer test
char-completion-transformer status
char-completion-transformer help
```

`build` validates source files, imports, training data, and required
directories; it does not build the Docker image or train the model. Add future
project dispatchers such as `rag` or `mcp` under `ai-model-dev/bin/` using
the same command contract.

## Setup

### Recommended: isolated VS Code Dev Container

The repository includes a CPU-first AI development container with Python 3.11,
PyTorch, Transformers, Safetensors, JupyterLab, IPython kernel support, and the
VS Code Python debugger. Nothing is installed into the host Python environment.
See **[DEVCONTAINER.md](../DEVCONTAINER.md)** for the complete step-by-step setup,
verification, usage, GPU, and troubleshooting guide.
See **[TEST.md](TEST.md)** for manual VS Code and terminal-only test procedures.

Create the host model directory once:

```bash
mkdir -p ../models
```

This creates `ai-model-dev/models`. Place Hugging Face-style model files there
(for example `*.safetensors`,
`tokenizer.json`, and `config.json`). Then open this folder in VS Code, install
the **Dev Containers** extension if needed, and run **Dev Containers: Reopen in
Container** from the Command Palette.

Inside the container:

- the repository is the VS Code workspace;
- host models are available at `/models` through `$MODEL_DIR`;
- `/models` is read-only, preventing accidental model modification;
- Hugging Face downloads use `ai-model-dev/cache/huggingface` rather than the
  host user's global cache;
- the interpreter and debugger use `/usr/local/bin/python` in the container.

Start JupyterLab when required:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

The default image installs CPU-only PyTorch so it works without NVIDIA runtime
configuration. To use a CUDA build, change `TORCH_INDEX_URL` in
`../.devcontainer/Dockerfile` to the appropriate PyTorch CUDA wheel index and
add `"runArgs": ["--gpus", "all"]` to
`../.devcontainer/devcontainer.json`.

### Host installation (legacy)

This machine has no `torch` and `python3-venv` is not installed (`python3 -m venv`
fails with `No module named pip`), so PyTorch was installed CPU-only into the user
site-packages:

```bash
pip3 install --user --index-url https://download.pytorch.org/whl/cpu torch
```

**Caution — do not use a plain `pip3 install --user torch`.** It pulls in a modern
`setuptools` under `~/.local`, which shadows the system `setuptools 19.7` that the
`ttgfsandbox` (TomTom Greenfield) toolchain pins, and breaks `import setuptools`
system-wide. If that happens, undo it with:

```bash
pip3 uninstall -y setuptools     # removes only the ~/.local copy
```

For a properly isolated environment instead, install the venv package first
(needs sudo), then work inside it:

```bash
sudo apt install python3.10-venv
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Model archives and advanced usage

When `--no-overwrite` is used, the existing model bundle is moved into
`out/archive/<timestamp>/`. That directory keeps `ckpt.pt`,
`combined_corpus.txt`, and `training_files.txt` together. The manifest
records each source path, byte size, and SHA-256 hash. The newly trained bundle
is then written directly under `out/`, so `python prompt.py` always loads the
newest model by default. Load any archived model explicitly with:

```bash
python prompt.py --checkpoint out/archive/<timestamp>/ckpt.pt
```

Before any archive or training begins, `train_all.py` compares the current data
against `out/combined_corpus.txt` and `out/training_files.txt`. If nothing
changed and `out/ckpt.pt` exists, it exits without modifying any files.

Advanced/direct workflow:

```bash
python3 make_data.py                      # writes data/corpus.txt (~386k chars)
python3 train.py --steps 2000 --top-k 40  # trains, checkpoints to out/ckpt.pt, samples
python3 train.py --sample-only --prompt "The river" --top-k 40
```

Train on your own text with `--data path/to/file.txt`.

Useful flags: `--n-layer --n-head --n-embd --block-size --batch-size --lr --dropout`,
and for sampling `--prompt --max-new-tokens --temperature --top-k`.

## Measured run (CPU, 4 layers, 128-dim, 2000 steps)

```
corpus: 385,869 chars | vocab: 34 | train/val tokens: 347,282/38,587
model: 0.81M parameters on cpu
step     0 | train 3.5452 | val 3.5452
step   400 | train 0.2935 | val 0.2961
step  2000 | train 0.2345 | val 0.2366     ~7 minutes total
```

Sample after training (prompt `"The"`):

```
The machine painted a rusty key!
The old sailor carried an empty notebook on a tuesday, so the old sailor
remembered the morning light.
Every student questioned the broken clock without a word.
```

Starting from random characters, the model has learned spelling, word boundaries,
capitalisation, punctuation and the grammar of the corpus — purely from
next-character prediction.

## Test a trained checkpoint

```bash
python3 test.py                          # all four evaluations
python3 test.py --only grammar           # just one section
python3 test.py --prompt "The river"     # free-form continuation
python3 test.py --temperature 1.2 --top-k 20 --only sweep
```

`test.py` reports:

1. **Perplexity** on train / val / a freshly generated corpus with an unseen seed.
   Measured: 1.265 / 1.267 / 1.266 -- flat across all three, so the model generalised
   rather than memorised.
2. **Grammaticality** -- generated sentences are regex-matched against the exact
   grammar in `make_data.py`. Measured: 88/89 (98.9%) at `temperature=0.8, top_k=40`.
3. **Novelty** -- how many generated sentences are *not* verbatim training lines.
   Measured: 91.4% novel.
4. **Temperature sweep** -- grammaticality falls from 26/26 at `temp=0.2` to
   15/21 at `temp=1.5`, where spelling starts to break ("letterrs", "afte").

## How it works

1. **Tokenize** — one token per character (`CharTokenizer`). Real LLMs use BPE
   subwords; the rest of the architecture is identical.
2. **Embed** — a token embedding plus a learned position embedding, summed into
   the *residual stream*.
3. **Blocks** — each block is `x = x + attn(norm(x))` then `x = x + mlp(norm(x))`:
   - **Causal self-attention** lets every position gather information from earlier
     positions only (that mask is what makes it a *language* model). One `qkv`
     matmul serves all heads; each head learns a different relationship.
   - **MLP** widens 4x, applies GELU, projects back — the per-position computation.
   - Pre-norm + residuals are what keep deep stacks trainable.
4. **Predict** — a final LayerNorm and a linear head produce logits over the
   vocabulary at *every* position at once, so one forward pass yields `block_size`
   training signals. Loss is cross-entropy against the input shifted by one.
5. **Generate** — feed the prompt, sample from the last position's distribution,
   append, repeat (`temperature` / `top_k` control the randomness).

Training details worth knowing: AdamW with weight decay on matrices but not on
biases/LayerNorm gains, linear LR warmup then cosine decay, gradient clipping at
1.0, and weight tying between the input embedding and the output head.

## Scaling up

The code path is the same one used at scale — what changes is size and data:
a BPE tokenizer, `n_layer`/`n_embd` in the tens/thousands, `block_size` in the
thousands, mixed precision, gradient accumulation and multi-GPU sharding. Modern
models also swap LayerNorm for RMSNorm, absolute positions for RoPE, and GELU for
SwiGLU — each a small, local edit to `model.py`.

## Tracking how the weights move

```bash
python train.py --track                       # tracks at --eval-every
python train.py --track --track-every 50      # finer sampling
```

Writes `out/track.json` and `out/training_trajectory.png`, and prints a per-tensor
verdict table at the end.

Two sampling rates, because the costs differ. Loss, learning rate and the pre-clip
gradient norm are recorded **every step** — all three are already computed by the
loop, so it is free (`clip_grad_norm_` returns the pre-clip norm). Per-tensor
statistics are recorded **every N steps**, since measuring the update requires
cloning the tracked weights around `opt.step()`.

The headline number is the **update-to-weight ratio**:

```
ratio = || w_after - w_before || / || w_before ||     for one optimizer step
```

| Ratio | Reading |
| --- | --- |
| above 1e-2 | learning rate too high — expect loss spikes |
| around 1e-3 | healthy |
| below 1e-4 | that tensor has stopped learning |

It catches problems loss alone hides, because a dead layer is masked by the others
still improving. The four panels are loss with the val-train gap, the ratio per
tensor coloured by role, weight-norm drift, and the per-step gradient norm against
the clip threshold.
