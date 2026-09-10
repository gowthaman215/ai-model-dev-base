# Manual Test Guide

This guide verifies the isolated AI environment manually. It covers both VS Code
Dev Containers and plain terminal usage. No host Python packages are installed
by either procedure.

## Test paths

- Use [VS Code Dev Containers](#test-with-vs-code-dev-containers) to verify the
  intended IDE workflow.
- Use [Docker from the terminal](#test-entirely-from-the-terminal) when you do
  not want to use an IDE.

## Expected system behavior

The tests should confirm:

- Python, PyTorch, Transformers, Safetensors, and Jupyter run in Docker.
- The repository is available inside the container.
- `ai-model-dev/models` appears as `/models` and is read-only.
- `ai-model-dev/cache/huggingface` is writable and persistent.
- Host Ubuntu Python, Qt, and C++ installations are not changed.
- CUDA is unavailable in the default CPU-only image; this is expected.

## Quick knowledge-to-model acceptance test

Use this short test after the container environment is running.

From a new host terminal, the shortest test path is:

```bash
cd ~/10718695/github/ai-model-dev
./ai-dev-container.sh verify
./ai-dev-container.sh shell
```

To test the project-command workflow:

```bash
./ai-dev-container.sh shell
char-completion-transformer build
char-completion-transformer status
char-completion-transformer train
char-completion-transformer prompt
char-completion-transformer test
```

Use `exit` to leave the shared AI container.

### 1. Add a knowledge source

Create or copy a non-empty UTF-8 text file anywhere under `data/`. For example:

```bash
cat > data/manual_test.txt <<'EOF'
The project codename is Blue River.
Blue River uses a character-level Transformer.
EOF
```

`train_all.py` discovers text files recursively. Binary, empty, and non-UTF-8
files are skipped.

### 2. Train a new model safely

```bash
python train_all.py --no-overwrite
```

Expected behavior:

- If data changed, the previous `ckpt.pt`, `combined_corpus.txt`, and
  `training_files.txt` move together to `out/archive/<timestamp>/`.
- The newest model bundle is written directly under `out/`.
- If data did not change, the command prints the following and changes nothing:

```text
Training data is unchanged; keeping the existing model and doing nothing.
```

Confirm the new bundle and its exact source-data record:

```bash
ls -lh out/ckpt.pt out/combined_corpus.txt out/training_files.txt
cat out/training_files.txt
```

The manifest should include `data/manual_test.txt` with its byte size and
SHA-256 hash.

### 3. Prompt the model repeatedly

```bash
python prompt.py
```

Try:

```text
prompt> The project codename is
prompt> Blue River uses
prompt> /quit
```

The model should produce continuations rather than question-and-answer responses.
Because this is a small probabilistic character model, it may not reproduce a
fact reliably after one occurrence. Repeat important patterns in the training
data and provide enough related examples when testing learned behavior.

### 4. Confirm unchanged data is a no-op

Without editing anything under `data/`, run:

```bash
python train_all.py --no-overwrite
```

Expected:

```text
Training data is unchanged; keeping the existing model and doing nothing.
```

Confirm that no new archive directory was created:

```bash
find out/archive -mindepth 1 -maxdepth 1 -type d -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort
```

Remove `data/manual_test.txt` afterward only if it was created solely for this
test. Removing it changes the training data, so a later `train_all.py
--no-overwrite` invocation will correctly train another model.

## Test with VS Code Dev Containers

### 1. Open the repository

Run on the Ubuntu host:

```bash
cd ~/10718695/github/ai-model-dev
code .
```

### 2. Reopen it in the container

1. Press `Ctrl+Shift+P` in VS Code.
2. Select **Dev Containers: Reopen in Container**.
3. Wait for the initial image build and extension installation to finish.
4. Confirm the lower-left corner says **Dev Container: AI / PyTorch
   (isolated)**.

The post-create verification should print `AI container is ready`.

### 3. Check the Python interpreter

Open a new VS Code terminal and run:

```bash
which python
python --version
```

Expected:

```text
/usr/local/bin/python
Python 3.11.x
```

### 4. Check the AI packages

```bash
python -c "import torch, transformers, safetensors; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Safetensors:', safetensors.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected: all three versions are printed and `CUDA: False` is reported.

### 5. Check model storage

```bash
echo "$MODEL_DIR"
ls -la /models
```

Expected: `$MODEL_DIR` is `/models`, and files placed in
`ai-model-dev/models` on the host appear here.

Confirm that the model mount is protected:

```bash
touch /models/test-file
```

Expected: the command fails with a read-only filesystem error.

### 6. Check the Hugging Face cache

```bash
echo "$HF_HOME"
touch "$HF_HOME/test-file"
ls -l "$HF_HOME/test-file"
rm "$HF_HOME/test-file"
```

Expected: the file can be created, listed, and removed. The cache maps to
`ai-model-dev/cache/huggingface` on the host.

### 7. Test the trained model

For the interactive prompt loop:

```bash
python prompt.py
```

Enter `/quit` to stop. To rebuild the model first from every UTF-8 text file
under `data/`, run:

```bash
python train_all.py
cat out/training_files.txt
```

The manifest shows exactly which files supplied training knowledge.

To preserve the current checkpoint and create a separate checkpoint for the new
training run:

```bash
python train_all.py --no-overwrite
find out/archive -maxdepth 2 -type f -print
cat out/training_files.txt
```

The complete old bundle is moved to `out/archive/<timestamp>/`, while the
newly trained bundle is written directly under `out/`. Each bundle contains:

```text
ckpt.pt
combined_corpus.txt
training_files.txt
```

The manifest records the SHA-256 hash and byte size of every source training
file. Load an archived run with:

```bash
python prompt.py --checkpoint out/archive/<timestamp>/ckpt.pt
```

Run the same training command again without changing anything under `data/`:

```bash
python train_all.py --no-overwrite
```

Expected:

```text
Training data is unchanged; keeping the existing model and doing nothing.
```

No checkpoint or archive timestamp should change.

```bash
python test.py --prompt "The river" --n-chars 300
```

Expected: text is generated using `out/ckpt.pt`.

Run all repository evaluations:

```bash
python test.py
```

The command evaluates perplexity, grammaticality, novelty, and sampling
temperature.

### 8. Test Jupyter

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Expected: Jupyter starts and VS Code offers the forwarded port `8888`. Stop the
server with `Ctrl+C`.

## Test entirely from the terminal

These steps use Docker directly and do not open VS Code.

### 1. Open the AI root directory

```bash
cd ~/10718695/github/ai-model-dev
mkdir -p models cache/huggingface
```

All subsequent commands in this section assume the current directory is
`ai-model-dev`.

### 2. Build the image

```bash
docker build \
  --tag ai-model-dev-base \
  --file .devcontainer/Dockerfile \
  .
```

Expected: the build finishes successfully and tags the image as
`ai-model-dev-base:latest`.

Confirm it exists:

```bash
docker image inspect ai-model-dev-base:latest >/dev/null \
  && echo "image exists"
```

### 3. Run the environment verification

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env HF_HOME=/ai-cache \
  --env MODEL_DIR=/models \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  --mount type=bind,source="$PWD/cache/huggingface",target=/ai-cache \
  --mount type=bind,source="$PWD",target=/workspace/ai-model-dev \
  --workdir /workspace/ai-model-dev/char-completion-transformer \
  ai-model-dev-base \
  python /workspace/ai-model-dev/.devcontainer/verify_environment.py
```

Expected output includes:

```text
AI container is ready
Python:       3.11.x
PyTorch:      ...+cpu
Transformers: ...
Safetensors:  ...
Models:       /models (host-mounted, read-only)
HF cache:     /ai-cache (host-mounted, writable)
CUDA:         False
```

The `--user` option maps the current enterprise Linux user and group into the
container. This keeps files created in the workspace and cache owned by the host
user rather than `root`.

### 4. Run a PyTorch smoke test

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  ai-model-dev-base \
  python -c "import torch; x=torch.tensor([1., 2., 3.]); print(x * 2); print('torch test passed')"
```

Expected:

```text
tensor([2., 4., 6.])
torch test passed
```

### 5. Run the trained model

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env HF_HOME=/ai-cache \
  --env MODEL_DIR=/models \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  --mount type=bind,source="$PWD/cache/huggingface",target=/ai-cache \
  --mount type=bind,source="$PWD",target=/workspace/ai-model-dev \
  --workdir /workspace/ai-model-dev/char-completion-transformer \
  ai-model-dev-base \
  python test.py --prompt "The river" --n-chars 300
```

Expected: the checkpoint generates a continuation beginning with `The river`.

Run the complete evaluation suite by replacing the final line with:

```text
python test.py
```

### 6. Confirm `/models` is read-only

```bash
docker run --rm \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  ai-model-dev-base \
  bash -c 'if touch /models/write-test 2>/dev/null; then echo "FAILED: mount is writable"; exit 1; else echo "PASSED: /models is read-only"; fi'
```

Expected:

```text
PASSED: /models is read-only
```

### 7. Confirm the cache is writable

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$PWD/cache/huggingface",target=/ai-cache \
  ai-model-dev-base \
  bash -c 'touch /ai-cache/write-test && rm /ai-cache/write-test && echo "PASSED: cache is writable"'
```

Expected:

```text
PASSED: cache is writable
```

### 8. Open an interactive container terminal

Use this when you want to run multiple commands manually:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env HF_HOME=/ai-cache \
  --env MODEL_DIR=/models \
  --publish 127.0.0.1:8888:8888 \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  --mount type=bind,source="$PWD/cache/huggingface",target=/ai-cache \
  --mount type=bind,source="$PWD",target=/workspace/ai-model-dev \
  --workdir /workspace/ai-model-dev/char-completion-transformer \
  ai-model-dev-base \
  bash
```

At the container prompt, try:

```bash
which python
python --version
python /workspace/ai-model-dev/.devcontainer/verify_environment.py
python test.py --prompt "The river" --n-chars 300
exit
```

`exit` closes and removes the temporary container. Files written to `/workspace`
or `/ai-cache` remain on the host because those directories are bind-mounted.
Port 8888 is published so Jupyter started later from this interactive shell is
reachable from the host browser.

### 9. Run Jupyter without an IDE

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env HF_HOME=/ai-cache \
  --env MODEL_DIR=/models \
  --publish 127.0.0.1:8888:8888 \
  --mount type=bind,source="$PWD/models",target=/models,readonly \
  --mount type=bind,source="$PWD/cache/huggingface",target=/ai-cache \
  --mount type=bind,source="$PWD",target=/workspace/ai-model-dev \
  --workdir /workspace/ai-model-dev/char-completion-transformer \
  ai-model-dev-base \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Copy the URL containing the access token from the terminal and open it in a web
browser. Binding to `127.0.0.1` prevents access from other network devices. Stop
Jupyter with `Ctrl+C`.

Use the `http://127.0.0.1:8888/lab?token=...` URL printed by Jupyter. Do not use
the `file:///tmp/...` URL: `/tmp` is inside the container and is not directly
accessible to the host browser. If port 8888 was not published when the
container was created, stop and recreate the container; Docker cannot add a port
mapping to an already-running container.

The prompt `I have no name!` is harmless in this terminal-only workflow. The
host uses a large enterprise user ID that is not present in the container's
`/etc/passwd`, while `--user "$(id -u):$(id -g)"` intentionally preserves host
ownership for bind-mounted files. VS Code Dev Containers updates the container
user automatically and normally displays the `vscode` username instead.

## Confirm the host environment was not modified

Run before and after the container tests on the host:

```bash
which python3
python3 --version
which qmake 2>/dev/null || true
which g++
```

The paths and versions should remain unchanged. Docker image creation and bind
mounting do not install Python packages into the host interpreter.

## Common failures

### Docker permission denied

If Docker reports permission denied for `/var/run/docker.sock`, ensure the user
has Docker access according to the host's administration policy. Do not change
system group membership on a managed corporate machine without approval.

### Bind source path does not exist

Run from `ai-model-dev`:

```bash
mkdir -p models cache/huggingface
```

### Port 8888 is already in use

Use another host port while leaving the container port unchanged:

```text
--publish 127.0.0.1:8889:8888
```

Then open the Jupyter URL using port `8889`.

### Stop a running terminal container

Press `Ctrl+C`, then type `exit` if a shell remains open. From another host
terminal, inspect running containers with:

```bash
docker ps
```
