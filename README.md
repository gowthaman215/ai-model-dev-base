# AI model development workspace

This repository contains one shared, isolated Python/PyTorch/Jupyter development
container and multiple independently operated AI projects.

```text
ai-model-dev/
├── .devcontainer/                 shared VS Code Dev Container
├── ai-dev-container.sh            host Docker launcher
├── bin/                            project commands available in the container
├── cache/huggingface/              persistent Hugging Face cache
├── models/                         persistent model files
├── char-completion-transformer/    character-level completion project
└── weight-visualizer/              checkpoint weight inspection and plots
```

## Start the environment

```bash
cd ~/10718695/github/ai-model-dev
./ai-dev-container.sh build
./ai-dev-container.sh shell
```

Inside the container:

```bash
char-completion-transformer build
char-completion-transformer train
char-completion-transformer prompt
char-completion-transformer test
char-completion-transformer status

weight-visualizer build
weight-visualizer run
weight-visualizer serve
weight-visualizer status
```

Start Jupyter directly from the host:

```bash
./ai-dev-container.sh jupyter
```

See [DEVCONTAINER.md](DEVCONTAINER.md) for environment setup,
[char-completion-transformer/README.md](char-completion-transformer/README.md)
for the model workflow, and
[weight-visualizer/README.md](weight-visualizer/README.md) for checkpoint
inspection.

Note: `matplotlib` was added to `.devcontainer/requirements-ai.txt` for
weight-visualizer. Run `./ai-dev-container.sh build` if your image predates it.

See [AGI-SYSTEM-ROADMAP.md](AGI-SYSTEM-ROADMAP.md) for the staged plan covering
model fundamentals, RAG, memory, agents, reasoning, evaluation, and safety.
