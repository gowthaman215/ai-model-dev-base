# Development container recovery — 2026-09-10

The shared `.devcontainer` directory was missing after moving the repository to
`~/10718695/github/ai-model-dev-base`. It was not present in the new repository's
Git history. The surviving Docker image and repository documentation were used
to reconstruct a functional configuration; these are not claimed to be
byte-for-byte copies of the deleted files.

## 1. Evidence used

Inspected locally:

```bash
docker image ls --no-trunc
docker history --no-trunc --format '{{.CreatedBy}}' ai-model-dev
docker image inspect ai-model-dev
```

Original image: `ai-model-dev:latest`, ID
`sha256:635f5749edf0e6a574cd656fcf7ffa994e0543fc7385c43466e7c9ae4ce58532`.

The image identified the Microsoft Python Dev Container base, variant
`3.11-bookworm`, Python 3.11.13, the `vscode` user, CPU PyTorch installation,
and an `ai-container` kernel named **Python (AI container)**. Direct dependency
versions were read from installed package metadata without downloading models.

## 2. Restored files

| File | Purpose |
| --- | --- |
| `.devcontainer/Dockerfile` | Microsoft Python 3.11 base, CPU PyTorch, AI packages, Jupyter kernel |
| `.devcontainer/requirements-ai.txt` | Dependency versions matching the surviving image |
| `.devcontainer/devcontainer.json` | Repository mount, model/cache mounts, project command PATH, extensions and port 8888 |
| `.devcontainer/verify_environment.py` | Dependency imports, CPU tensor operation, mount permissions and command checks |
| `.dockerignore` | Restricts build context to Dockerfile and requirements; excludes checkpoints and caches |

PyTorch is pinned separately as `TORCH_VERSION=2.14.0` and uses the CPU index.
The requirements include Transformers 5.16.1, Safetensors 0.8.0, Matplotlib
3.11.1, NumPy 2.4.6, JupyterLab 4.6.3, IPykernel 7.3.0 and debugpy 1.8.21.
Direct versions are pinned; the base-image tag and transitive dependencies can
still change on future builds, so this is not a complete dependency lockfile.

## 3. Paths after the move

| Purpose | Host | Container |
| --- | --- | --- |
| Repository | `~/10718695/github/ai-model-dev-base` | `/workspace/ai-model-dev` |
| Local models | `models/` inside the repository | `/models` (read-only mount) |
| Hugging Face cache | `cache/huggingface/` inside the repository | `/ai-cache` (writable) |

The launcher discovers its own location, so the host rename requires no alias
or bashrc change. Existing internal paths remain unchanged. New builds and launcher commands use
`ai-model-dev-base:latest`; `ai-model-dev:latest` above identifies the original
image used as recovery evidence.
VS Code opens the repository root; the host shell launcher starts in
`char-completion-transformer`. Both project commands are available on PATH.

## 4. Verify using the existing image

On the host:

```bash
cd ~/10718695/github/ai-model-dev-base
./ai-dev-container.sh verify
./ai-dev-container.sh shell
```

Expected verification ends with `AI container is ready`. It checks that `/models`
is read-only and that the cache is writable without modifying model files.
Inside the shell:

```bash
char-completion-transformer help
weight-visualizer help
weight-visualizer build
```

`char-completion-transformer build` also needs training data in its `data/`
directory. Training and evaluation require restoring any data/checkpoints not
copied during the repository move; environment recovery does not regenerate them.

## 5. Rebuild the image

Normal rebuild (updates the image used by the launcher):

```bash
./ai-dev-container.sh build
./ai-dev-container.sh verify
```

To test the Dockerfile under a separate tag, retaining the original image:

```bash
docker build --tag ai-model-dev-base:recovery-check --file .devcontainer/Dockerfile .
```

## 6. Jupyter and VS Code

From the host:

```bash
./ai-dev-container.sh jupyter
```

Open the tokenized localhost URL printed by Jupyter. Stop it with Ctrl+C.
Use `AI_JUPYTER_PORT=8889 ./ai-dev-container.sh jupyter` if host port 8888
is occupied. Jupyter and the weight explorer share port 8888 by default and
should not be launched simultaneously on that port.

For VS Code, open the **repository root**, then run **Dev Containers: Reopen in
Container**. The configuration creates mount-source directories before startup,
uses the `vscode` user with host UID adjustment, forwards port 8888 and runs the
verification script after creation. Select **Python (AI container)** for notebooks.

## 7. Preserve the recovery in Git

Review the new files with `git status --short` before committing. `.devcontainer`
is a hidden directory; include hidden files when moving or backing up repositories.
No commit is made automatically as part of this recovery.
