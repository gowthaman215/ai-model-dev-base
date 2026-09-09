# Isolated AI Development Environment

This guide recreates and operates the VS Code Dev Container used by this
repository. Python, PyTorch, Transformers, Safetensors, Matplotlib, Jupyter, and
the Python debugger run inside Docker. The Ubuntu host's Python, Qt, and C++ installations
are not modified.

For a test-focused checklist, including operation without an IDE, see
**[TEST.md](TEST.md)**.

## Architecture

```text
VS Code
  -> Dev Containers extension
    -> Docker container
       - Python 3.11
       - CPU-only PyTorch
       - Transformers and Safetensors
       - Matplotlib (plots for weight-visualizer and training tracking)
       - JupyterLab and IPython kernel
       - debugpy / VS Code Python debugger
    -> read-only bind mount: ai-model-dev/models -> /models
    -> bind mount: ai-model-dev/cache/huggingface -> ~/.cache/huggingface

Ubuntu host Python / Qt / C++: unchanged
```

## 1. Prerequisites

Install these on the host:

1. Docker Engine or Docker Desktop.
2. Visual Studio Code.
3. The VS Code extension **Dev Containers** (`ms-vscode-remote.remote-containers`).

Confirm Docker is available:

```bash
docker --version
docker run --rm hello-world
```

## 2. Prepare the host model directory

Create the directory once on the Ubuntu host:

```bash
cd ~/10718695/github/ai-model-dev
mkdir -p models cache/huggingface
```

Copy local Hugging Face model files into it. A typical directory may contain:

```text
ai-model-dev/models/my-model/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors
```

The Dev Container mounts `ai-model-dev/models` at `/models` in read-only mode.
It mounts `ai-model-dev/cache/huggingface` as the container user's Hugging Face
cache. Both directories survive image rebuilds without touching global host
configuration.

## 3. Files used by the setup

The repository contains:

```text
.devcontainer/
  devcontainer.json       VS Code settings, extensions, ports, and mounts
  Dockerfile              Python base image and package installation
  requirements-ai.txt     AI and notebook Python dependencies
  verify_environment.py   import and mount checks after container creation
```

The default image uses the PyTorch CPU wheel index. This makes it portable to
hosts that do not have an NVIDIA GPU or NVIDIA Container Toolkit.

## 4. Open the project in the container

From a terminal on the host:

```bash
cd ~/10718695/github/ai-model-dev
code .
```

In VS Code:

1. Open the Command Palette with `Ctrl+Shift+P`.
2. Select **Dev Containers: Reopen in Container**.
3. Wait for the image to build and the extensions to install.
4. Check the terminal output from `verify_environment.py`.

The first build downloads the base image and Python packages. Later builds use
Docker's layer cache unless the Dockerfile or dependency list changes.

## 5. Verify isolation and installed tools

Run these commands in the VS Code terminal after it reopens in the container:

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, safetensors, matplotlib; print(transformers.__version__, safetensors.__version__, matplotlib.__version__)"
echo "$MODEL_DIR"
ls -la /models
```

Expected results:

- `which python` reports `/usr/local/bin/python`.
- `$MODEL_DIR` is `/models`.
- the model files from the host are visible under `/models`.
- `torch.cuda.is_available()` is `False` for the default CPU image.

Confirm that the mount is protected from writes:

```bash
touch /models/write-test
```

The command should fail with a read-only filesystem error. This is intentional.

## 6. Load a local Transformers model

Replace `my-model` with the directory name copied into `ai-model-dev/models`:

```python
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.path.join(os.environ["MODEL_DIR"], "my-model")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
)
```

Using `local_files_only=True` prevents an accidental network download when a
required local file is missing.

## 7. Run the repository scripts

In the container terminal:

```bash
cd char-completion-transformer
python make_data.py
python train.py --steps 2000 --top-k 40
python test.py
```

Repository outputs are written to the mounted VS Code workspace, so they remain
available on the host after the container stops.

## 8. Use Jupyter

Start JupyterLab inside the container:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

VS Code forwards port 8888 and shows a notification. For notebooks opened
directly in VS Code, select the kernel **Python (AI container)**.

Stop JupyterLab with `Ctrl+C` in its terminal.

## 9. Debug Python in VS Code

1. Open a Python file.
2. Add a breakpoint by clicking beside a line number.
3. Press `F5`.
4. Choose **Python File** if VS Code asks for a debug configuration.

The debugger and Python extension are installed inside the container, and use
the container interpreter rather than the host interpreter.

## 10. Add or change Python dependencies

Edit `.devcontainer/requirements-ai.txt`, then rebuild:

1. Open the Command Palette.
2. Select **Dev Containers: Rebuild Container**.

Keep PyTorch out of `requirements-ai.txt`; the Dockerfile installs it separately
from the CPU wheel index.

## 11. Rebuild or remove container resources

Rebuild after changing the Dockerfile or dependencies:

```text
Ctrl+Shift+P -> Dev Containers: Rebuild Container
```

List related Docker resources on the host:

```bash
docker image ls
docker volume ls
docker ps -a
```

The cache lives inside `ai-model-dev`. Clear its contents only when downloaded
artifacts are no longer needed:

```bash
rm -rf ~/10718695/github/ai-model-dev/cache/huggingface/*
```

This deletion is permanent for downloaded cache files, but does not affect
`ai-model-dev/models`. Review the path carefully before running the command.

## 12. Current system hardware assessment

Measured on 9 September 2026:

| Component | Detected hardware | Assessment |
|---|---|---|
| CPU | AMD Ryzen AI 7 350, 8 cores / 16 threads, up to 4.31 GHz | Good for Python development, preprocessing, notebooks, and small CPU training jobs |
| CPU instructions | AVX2, AVX-512, AVX-512 BF16 and VNNI available | Useful acceleration for supported CPU inference/training libraries |
| RAM | 14 GiB visible to Linux (approximately 16 GB installed) | Adequate for small models; restrictive for large-model training or inference |
| Swap | 2 GiB configured; fully used during inspection | Current memory pressure is high; close memory-heavy applications before model work |
| GPU | Integrated AMD Radeon 860M; no `nvidia-smi` or NVIDIA CUDA GPU detected | The default CPU container is the reliable configuration |
| Storage | 468 GiB filesystem, 182 GiB free | Adequate for the container, datasets, and several small/medium model files |
| Docker | Docker 29.7.2 | Meets the container requirement |
| Architecture | x86-64 with AMD-V virtualization | Compatible with the selected Dev Container image |

### What this machine is suitable for

- Developing and debugging PyTorch and Transformers code.
- Jupyter notebooks, tokenization, preprocessing, and dataset inspection.
- Training this repository's approximately 0.8-million-parameter transformer.
- Training small experimental models, generally up to tens of millions of
  parameters, with conservative batch and sequence sizes.
- CPU inference with small models. Quantized models can reduce RAM use, but
  inference will still be much slower than on a discrete accelerator.
- Fine-tuning only small models or very small parameter subsets. Use a remote GPU
  for practical LoRA/QLoRA work on multi-billion-parameter models.

### What this machine is not suitable for

- Training a modern multi-billion-parameter language model from scratch.
- Fast inference with large language, image-generation, or multimodal models.
- CUDA-dependent workflows: there is no NVIDIA CUDA device.
- Keeping many model copies in memory. Model weights are only part of memory
  usage; training also needs gradients, optimizer state, activations, and data.

### Practical limits and recommendations

1. Keep the CPU-only PyTorch configuration unless an independently verified AMD
   GPU runtime is added later.
2. Before training, use `free -h` and make sure several GiB of RAM is available.
   At inspection time only 4.5 GiB was available and swap was full.
3. Start with batch sizes of 1-8 and short sequence lengths, then increase them
   while watching memory.
4. Use `float32` for maximum CPU compatibility. BF16 support exists at the CPU
   instruction level, but actual benefit depends on each PyTorch operation.
5. Keep at least 30-50 GiB free for Docker layers, caches, datasets, checkpoints,
   and temporary files. The detected 182 GiB free space is currently healthy.
6. Use a cloud or workstation GPU with substantial VRAM for billion-parameter
   training or responsive large-model inference.

Re-run this inventory later from the host:

```bash
lscpu
free -h
df -h ~
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
nvidia-smi
```

Hardware being sufficient does not guarantee a model will fit. Check the model's
parameter count, numeric precision, context length, batch size, optimizer, and
framework overhead before starting a large job.

## 13. Optional NVIDIA GPU support

First install and verify the NVIDIA driver and NVIDIA Container Toolkit on the
host. Confirm that Docker can access the GPU before changing this project:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Then make two configuration changes:

1. In `.devcontainer/Dockerfile`, change `TORCH_INDEX_URL` from the CPU index to
   the CUDA wheel index compatible with the installed driver.
2. In `.devcontainer/devcontainer.json`, add this top-level property:

```json
"runArgs": ["--gpus", "all"]
```

Rebuild the container and verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do not enable `--gpus all` on a machine without a working NVIDIA Container
Toolkit; the container will fail to start.

## 14. Troubleshooting

### `/models` does not exist or the container fails to start

Create the source directory on the host and rebuild:

```bash
cd ~/10718695/github/ai-model-dev
mkdir -p models cache/huggingface
```

### Model changes are rejected

The mount is deliberately read-only. Write converted or fine-tuned models into
the repository workspace or another writable container volume, then copy the
finished files into `ai-model-dev/models` from the host.

### Imports fail after changing requirements

Use **Dev Containers: Rebuild Container**. Installing interactively with `pip`
changes only the current container and those changes disappear when it is
recreated.

### Host tools appear in the terminal

Confirm the lower-left corner of VS Code says **Dev Container: AI / PyTorch
(isolated)** and run `which python`. If necessary, use **Dev Containers: Reopen
in Container** again.

### Reset only the Hugging Face download cache

Close the Dev Container, inspect the target, then remove its contents on the host:

```bash
ls -la ~/10718695/github/ai-model-dev/cache/huggingface
rm -rf ~/10718695/github/ai-model-dev/cache/huggingface/*
```

The cache directory remains in place and will be populated again as needed.

## 15. Where all AI files live

The complete layout is contained beneath `ai-model-dev`:

```text
ai-model-dev/
  .devcontainer/              shared Docker and VS Code environment
  DEVCONTAINER.md             this guide
  models/                    persistent local model files
  cache/huggingface/         persistent model download cache
  char-completion-transformer/
    README.md
    model.py
    train.py
    test.py
```
