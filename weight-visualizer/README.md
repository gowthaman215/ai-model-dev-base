# Weight visualizer

Reads a transformer checkpoint and plots where its parameters live, how their
values are distributed, and how those statistics change with depth.

Everything runs inside the shared container. Model downloads go to
`HF_HOME=/ai-cache`, which is bound to `../cache/huggingface`, so nothing is
written outside this workspace.

## Usage

```bash
cd ~/10718695/github/ai-model-dev
./ai-dev-container.sh shell
```

Inside the container:

```bash
weight-visualizer build                              # validate deps
weight-visualizer run                                # default: gpt2 (124M)
weight-visualizer run --model Qwen/Qwen2.5-0.5B
weight-visualizer run --model /models/my-checkpoint  # a local checkpoint
weight-visualizer run --model gpt2 --view-mode anatomy   # SVG only
weight-visualizer status                             # list plots, cache size
weight-visualizer clean                              # delete plots/
```

Output lands in `plots/`.

## What each plot shows

| File | Shows |
| --- | --- |
| `01_parameter_budget.png` | Parameter count by role — embedding / attention / FFN / norm. On GPT-2, `wte` alone is 28% of the model. |
| `02_distributions.png` | Histogram of weight values per role, log counts. Roughly Gaussian and centred on zero, with heavy tails. |
| `03_depth_trend.png` | Weight spread and outlier magnitude per layer. Shows how statistics drift from the first block to the last. |
| `04_heatmaps.png` | A 128x128 crop of a real matrix from each role. Deliberately included to show it looks like static — meaning lives in superposition across directions, not in single entries. |
| `anatomy.svg` | The model drawn to scale. Left panel: the whole model as a column, area proportional to parameter count, so the embedding table can be weighed against the entire layer stack. Right panel: one transformer block with every matrix at its true aspect ratio on a shared px-per-element scale. Hover any block for its tensor name, shape and statistics. |
| `05_outlier_channels.png` | Per-channel max absolute weight. The spikes are the outlier channels that break naive quantization. |

## Checkpoint formats

Both safetensors and PyTorch `.pt` / `.pth` / `.ckpt` / `.bin` are read through
`loader.Checkpoint`. Pass a Hugging Face repo id, a directory of `*.safetensors`,
or a path to a `.pt` file.

The difference is memory. safetensors is memory-mapped, so tensors and
sub-rectangles are read lazily off disk. A `.pt` is a pickle — there is no way to
read part of one, so the whole state dict is loaded once and held in memory. Fine
for small checkpoints; for a multi-GB `.pt`, convert first and the lazy path
returns:

```python
from loader import Checkpoint
Checkpoint("out/ckpt.pt").to_safetensors("out/ckpt.safetensors")
```

Nested state dicts are unwrapped automatically — the loader looks under `model`,
`state_dict`, `model_state_dict`, `net` and `weights` before falling back to
treating the top level as the state dict itself.

## Live explorer

```bash
weight-visualizer serve --model gpt2
weight-visualizer serve --model Qwen/Qwen2.5-0.5B --port 8888
```

Then open <http://127.0.0.1:8888>. Scroll to zoom, drag to pan, click any cell for
its exact value. Pick any 2D tensor in the checkpoint from the dropdown — including
the full 50257x768 embedding table.

Nothing is precomputed and nothing is written to disk. `safetensors` slices lazily,
so a request for rows 400-600 of a 4864x896 matrix reads only those rows off the
file. The browser asks for the window it is currently showing at a stride matched
to the zoom level:

| Zoom | What you get |
| --- | --- |
| below 1x | server strides the read — a sampled view, stride shown in the panel |
| 1x and above | every cell is an exact weight |
| above 26x | each cell also prints its own value |

A click always issues a separate single-element read, so the pinned number is the
true value from the checkpoint regardless of the stride in effect. Verified against
a direct read: `h.0.mlp.c_fc.weight[100,200]` serves `0.39199987053871155`, which
matches `safe_open(...).get_slice(...)` exactly.

The server is stdlib-only (`http.server`) — no Flask, nothing added to the image.
Stop it with ctrl-c, or `docker stop $(docker ps -q --filter ancestor=ai-model-dev-base)`
from the host.

## View modes

`--view-mode` selects what a run emits: `plots` for the five PNG plates, `anatomy`
for the scalable vector map, `all` for both (the default). To compare several
models at one scale, pass the same `--canvas-width` to each run — the value comes
from `anatomy.block_canvas_width()` for the widest model:

```bash
weight-visualizer run --model gpt2         --view-mode anatomy --canvas-width 647.2
weight-visualizer run --model gpt2-medium  --view-mode anatomy --canvas-width 647.2
```

Without it each SVG sizes to its own content and areas are no longer comparable
between files.

## Non-parameter buffers

Some checkpoints store registered buffers alongside real parameters. GPT-2 keeps
its causal attention mask as a `float32` tensor named `h.N.attn.bias` with shape
`(1, 1, 1024, 1024)` — 1.05M elements per layer. Counting those inflates both the
total and the attention bucket (gpt2 reads as 137M instead of its real 124M), so
`BUFFER_RE` skips them and the run reports how many were dropped. The patterns
anchor tightly, because `attn.c_attn.bias` on the same module *is* a real
parameter.

Verified totals after skipping: gpt2 124.4M, gpt2-medium 354.8M,
Qwen2.5-0.5B 494.0M — all matching published figures.

A second naming bug affected the depth plate: `LAYER_RE` required a dot before
the layer keyword, but GPT-2 tensor names begin at `h.0.` with no leading dot, so
every GPT-2 tensor got `layer: None` and the depth trend rendered empty. The
pattern is now anchored `(?:^|\.)`.

## Memory behaviour

The script reads one tensor at a time via `safetensors.safe_open` and keeps only
summary statistics plus a subsample of values, so peak memory is roughly the
size of the largest single tensor, not the model. That matters on this host,
which has ~14 GB RAM total. Small models (gpt2, Qwen2.5-0.5B) are the sensible
targets here; a 7B checkpoint will read fine but downloads ~15 GB. A `.pt` is the
exception — it is loaded whole, so its size is your memory floor.

The anatomy map's px-per-element scale auto-fits by default, so a 0.8M-parameter
char model and a 494M one are both legible standalone. Pin `--px-scale` or
`--canvas-width` when several models must stay comparable to each other.

## Adding a model

Either pass a Hugging Face repo id (downloaded into the shared cache), or drop a
checkpoint directory containing `*.safetensors` into `../models/` and pass
`--model /models/<name>`. `models/` is mounted read-only in the container.

## Dependencies

Needs `matplotlib`, which was added to `.devcontainer/requirements-ai.txt`.
If the image predates that change, rebuild it:

```bash
./ai-dev-container.sh build
```

Checkpoints in `.bin`-only repos are not supported — safetensors only.
