# AI Coding CLI — Ubuntu Installation & Run Command Reference

> Verified against official documentation in August 2026 where available.
> Commands and product names can change over time, so re-check official documentation before using this as long-term automation input.

## Purpose

This document provides a clean Ubuntu/Linux reference for installing and running major AI coding CLIs and local-model tools.

Covered tools:

- OpenAI Codex CLI
- Anthropic Claude Code
- Google Gemini CLI
- Google Antigravity CLI
- xAI Grok Build
- GitHub Copilot CLI
- Cursor CLI / Cursor Agent
- AWS Kiro CLI
- Aider
- Ollama

---

# 1. Quick Comparison

| Provider | CLI / Product | Install Command | Run Command |
|---|---|---|---|
| OpenAI | Codex CLI | `npm install -g @openai/codex` | `codex` |
| Anthropic | Claude Code | `curl -fsSL claude.ai/install.sh \| bash` | `claude` |
| Google | Gemini CLI | `npm install -g @google/gemini-cli` | `gemini` |
| Google | Antigravity CLI | `curl -fsSL https://antigravity.google/cli/install.sh \| bash` | `agy` |
| xAI | Grok Build | `curl -fsSL https://x.ai/cli/install.sh \| bash` | `grok` |
| GitHub | Copilot CLI | `npm install -g @github/copilot` | `copilot` |
| Cursor | Cursor CLI / Agent | `curl https://cursor.com/install -fsS \| bash` | `agent` |
| AWS | Kiro CLI | `curl -fsSL https://cli.kiro.dev/install \| bash` | `kiro` |
| Open source | Aider | `curl -LsSf https://aider.chat/install.sh \| sh` | `aider` |
| Ollama | Local model runtime | `curl -fsSL https://ollama.com/install.sh \| sh` | `ollama` / `ollama run <model>` |

---

# 2. Recommended Ubuntu Prerequisites

Before installing npm-based CLIs, check Node.js and npm:

```bash
node --version
npm --version
```

Some current requirements differ by tool:

```text
Gemini CLI        Node.js 20+
GitHub Copilot    Node.js 22+
```

Using a recent Node.js LTS release is usually the simplest approach.

Also check:

```bash
git --version
curl --version
```

For Python-based tools:

```bash
python3 --version
pip3 --version
```

---

# 3. OpenAI Codex CLI

## Install

```bash
npm install -g @openai/codex
```

Short form:

```bash
npm i -g @openai/codex
```

## Verify

```bash
codex --version
```

## Run

```bash
codex
```

Run it from a repository:

```bash
cd ~/projects/my-project
codex
```

## Non-interactive / automation usage

Codex also supports command execution workflows such as:

```bash
codex exec "Review this repository and identify likely bugs."
```

## Typical prompts

```text
Explain the architecture of this repository.

Review the current git diff.

Find possible memory leaks.

Trace the call flow for this function.

Find the likely root cause of this bug.
```

## Authentication

Start:

```bash
codex
```

Then follow the ChatGPT/OpenAI sign-in flow or configured API-key workflow.

## Official source

```text
https://openai.com/codex/
```

---

# 4. Anthropic Claude Code

## Recommended native install

```bash
curl -fsSL claude.ai/install.sh | bash
```

## Alternative npm install

```bash
npm install -g @anthropic-ai/claude-code
```

Do not use:

```bash
sudo npm install -g @anthropic-ai/claude-code
```

because Anthropic warns that this can create permission/security problems.

## Verify / diagnose

```bash
claude doctor
```

## Run

```bash
claude
```

Run from project:

```bash
cd ~/projects/my-project
claude
```

## Update

```bash
claude update
```

Claude Code normally also checks for updates automatically.

## Typical prompts

```text
Analyze this codebase.

Review the current changes.

Find the root cause of this crash.

Implement this Jira requirement.

Write tests for this component.
```

## Official documentation

```text
https://docs.anthropic.com/
```

---

# 5. Google Gemini CLI

## Requirement

Current official documentation requires:

```text
Node.js 20+
Ubuntu 20.04+
```

Check:

```bash
node --version
```

## Install

```bash
npm install -g @google/gemini-cli
```

## Run without installing

```bash
npx @google/gemini-cli
```

## Verify

```bash
gemini --version
```

## Run

```bash
gemini
```

From a repository:

```bash
cd ~/projects/my-project
gemini
```

## Update

```bash
npm install -g @google/gemini-cli@latest
```

## Preview channel

```bash
npm install -g @google/gemini-cli@preview
```

## Nightly channel

```bash
npm install -g @google/gemini-cli@nightly
```

## Select a model

Inside Gemini CLI:

```text
/model
```

Or when supported:

```bash
gemini -m <model-name>
```

Example pattern:

```bash
gemini -m gemini-3.1-pro-preview
```

Availability depends on account and rollout.

## Sandboxed execution

Example:

```bash
gemini --sandbox -y -p "Analyze this repository."
```

## Uninstall

```bash
npm uninstall -g @google/gemini-cli
```

## Official documentation

```text
https://github.com/google-gemini/gemini-cli
```

---

# 6. Google Antigravity CLI

Antigravity CLI is Google's newer terminal agent interface.

## Install on Linux

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
```

The executable is installed under:

```text
~/.local/bin/agy
```

## Ensure PATH is configured

If needed:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Verify

```bash
agy --help
```

## Run

```bash
agy
```

## Authentication

On a normal local Ubuntu machine:

```bash
agy
```

If no stored credentials exist, Antigravity opens the browser for authentication.

For SSH sessions, it can provide a URL/code-based authentication flow.

## Gemini API-key mode

Create/edit:

```text
~/.gemini/antigravity-cli/settings.json
```

Example:

```json
{
  "modelProvider": "gemini"
}
```

Then:

```bash
export GEMINI_API_KEY="your-api-key"
agy
```

Important: according to Google's documentation, setting only `GEMINI_API_KEY` is not enough; `modelProvider` must also be configured to `gemini`.

## Official documentation

```text
https://antigravity.google/docs/cli/
```

---

# 7. xAI Grok Build

xAI's terminal coding agent is called Grok Build.

## Install

```bash
curl -fsSL https://x.ai/cli/install.sh | bash
```

## Alternative npm installation

```bash
npm install -g @xai-official/grok
```

## Verify

```bash
grok --help
```

## Run

```bash
grok
```

Running `grok` without arguments opens the interactive terminal UI.

## Diagnose installation

```bash
grok doctor
```

If supported by the installed release:

```bash
grok doctor fix
```

## Run from project

```bash
cd ~/projects/my-project
grok
```

## Typical prompts

```text
Analyze this C++ repository.

Review the current git changes.

Find the root cause of this crash.

Create unit tests for this module.
```

## Official documentation

```text
https://x.ai/build
https://docs.x.ai/build/
```

---

# 8. GitHub Copilot CLI

## Requirement

Current GitHub documentation requires:

```text
Node.js 22+
```

Check:

```bash
node --version
```

## Install

```bash
npm install -g @github/copilot
```

## Verify

```bash
copilot --version
```

## Run

```bash
copilot
```

## First-time authentication

Start:

```bash
copilot
```

Then inside the CLI:

```text
/login
```

Follow the GitHub authentication flow.

## Run from repository

```bash
cd ~/projects/my-project
copilot
```

## Non-interactive prompt

```bash
copilot -p "Explain this repository."
```

Example:

```bash
copilot -p "Review the current git changes for possible bugs."
```

## GitHub Actions usage

For automation, GitHub supports environment-based authentication such as:

```text
COPILOT_GITHUB_TOKEN
```

Example pattern:

```bash
copilot -p "Review today's git changes." --no-ask-user
```

Configure permissions carefully before using unattended write or shell access.

## Official documentation

```text
https://docs.github.com/en/copilot/
```

---

# 9. Cursor CLI / Cursor Agent

Cursor's terminal agent is installed using Cursor's installer.

## Install

```bash
curl https://cursor.com/install -fsS | bash
```

## PATH

If necessary:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Verify

Current documentation uses:

```bash
agent --version
```

Some older Cursor documentation/releases may refer to:

```bash
cursor-agent --version
```

## Run

Preferred current command:

```bash
agent
```

Older installations may use:

```bash
cursor-agent
```

## Run with initial prompt

```bash
agent "Review this repository."
```

## Ask mode

```bash
agent --mode=ask
```

## Plan mode

```bash
agent --mode=plan
```

## Non-interactive mode

```bash
agent -p "Review the current git diff."
```

## Choose model

```bash
agent --model <model-name>
```

Inside the interactive session:

```text
/model
```

Cursor exposes multiple model providers depending on the current plan/catalog.

## Resume

```bash
agent resume
```

List sessions:

```bash
agent ls
```

## Update

```bash
agent update
```

## Authentication

Depending on release:

```bash
agent login
```

Older builds may use:

```bash
cursor-agent login
```

## API key for automation

```bash
export CURSOR_API_KEY="your-api-key"
agent -p "Analyze this codebase."
```

## Official documentation

```text
https://cursor.com/cli
https://docs.cursor.com/en/cli/
```

---

# 10. AWS Kiro CLI

Kiro is AWS's agentic software-development environment with CLI support.

## Install

```bash
curl -fsSL https://cli.kiro.dev/install | bash
```

## Verify

```bash
kiro --help
```

## Run interactively

```bash
kiro
```

Kiro documentation also refers to project-associated chat sessions through:

```bash
kiro chat
```

## Run headlessly

```bash
kiro --print "Analyze this repository and identify likely defects."
```

## Example CI workflow command

```bash
kiro --print "Look at the latest CI failure logs, find the root cause, and apply a fix."
```

## Model selection

Inside Kiro:

```text
/model
```

## Useful commands

```text
/help
/model
/usage
/save
/load
/quit
```

## Official documentation

```text
https://kiro.dev/cli/
https://kiro.dev/docs/
```

---

# 11. Aider

Aider is an open-source, model-provider-neutral AI pair-programming CLI.

It can work with models from multiple providers.

## Recommended quick installation

```bash
curl -LsSf https://aider.chat/install.sh | sh
```

## Alternative installer

```bash
python -m pip install aider-install
aider-install
```

## Alternative uv installation

```bash
python -m pip install uv
uv tool install --force --python python3.12 --with pip aider-chat@latest
```

## Verify

```bash
aider --version
```

## Run

```bash
cd ~/projects/my-project
aider
```

## Run with a provider/model

Example pattern:

```bash
aider --model <model-name>
```

API keys can be supplied through the environment or Aider configuration.

Examples of environment variables commonly used by providers:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Then:

```bash
aider --model <model-name>
```

## Why Aider is different

Aider itself is not the foundation model.

Architecture:

```text
Aider CLI
   ↓
Selected LLM provider
   ├── OpenAI
   ├── Anthropic
   ├── Gemini
   ├── OpenRouter
   ├── Local model
   └── Other supported providers
   ↓
Your repository
```

## Official documentation

```text
https://aider.chat/
https://aider.chat/docs/
```

---

# 12. Ollama

Ollama is primarily a local/cloud model runtime rather than a full coding agent.

It is useful for running models locally and connecting them to coding agents.

## Install on Ubuntu/Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Verify

```bash
ollama -v
```

## Open interactive menu

```bash
ollama
```

## Start server manually

```bash
ollama serve
```

## Run a model

Pattern:

```bash
ollama run <model>
```

Example:

```bash
ollama run llama3.2
```

Use the current Ollama model library to choose a suitable model.

## List installed models

```bash
ollama list
```

## Pull a model

```bash
ollama pull <model>
```

## Remove a model

```bash
ollama rm <model>
```

## Update Ollama on Linux

Run the installer again:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Service status

```bash
sudo systemctl status ollama
```

Start service:

```bash
sudo systemctl start ollama
```

## AMD GPU note

Ollama supports AMD GPU setups, but compatible ROCm/driver support depends on GPU and Linux configuration.

## Official documentation

```text
https://docs.ollama.com/
```

---

# 13. Meta / Llama CLI Clarification

Meta does not provide a direct general-purpose coding CLI equivalent to:

```text
codex
claude
gemini
grok
copilot
agent
kiro
```

Instead, Meta provides Llama foundation models and tooling.

A common way to use Llama from Ubuntu terminal is:

```text
Llama
  ↓
Ollama / llama.cpp / inference server
  ↓
Aider or another agent
```

Example:

```bash
ollama run llama3.2
```

For coding-agent behavior, connect the local model to a tool such as Aider when supported.

---

# 14. Interactive vs Headless Usage

## Interactive

Best for development sessions:

```bash
codex
claude
gemini
agy
grok
copilot
agent
kiro
aider
```

## Headless / automation

Useful for:

```text
GitHub Actions
CI/CD
Automated code reviews
Jira triage
Log analysis
Documentation generation
Scheduled repository analysis
```

Examples:

```bash
codex exec "Review this repository."
```

```bash
copilot -p "Review this change."
```

```bash
agent -p "Review this change."
```

```bash
kiro --print "Analyze the build failure."
```

Exact security/approval flags vary between products.

---

# 15. Recommended Directory Workflow

For any repository:

```bash
cd ~/projects/my-project
```

Then start the desired agent:

```bash
codex
```

or:

```bash
claude
```

or:

```bash
gemini
```

or:

```bash
agy
```

or:

```bash
grok
```

or:

```bash
copilot
```

or:

```bash
agent
```

or:

```bash
kiro
```

or:

```bash
aider
```

---

# 16. Useful First Prompts for C++ / Qt Projects

```text
Give me a high-level architecture overview of this repository.
```

```text
Identify the main C++ components and their responsibilities.
```

```text
Find all Qt signal/slot connections related to this class.
```

```text
Review the current git diff for correctness and regressions.
```

```text
Trace the call flow for this function.
```

```text
Find possible memory leaks, dangling pointers, and lifetime problems.
```

```text
Analyze the thread-safety of this component.
```

```text
Find the most likely root cause of this issue.
```

```text
Do not modify code. Analyze only.
```

```text
Create an implementation plan before making changes.
```

---

# 17. Suggested AI CLI Selection

## Deep coding / repository analysis

Consider:

```text
Claude Code
Codex CLI
Cursor Agent
Gemini CLI / Antigravity
Grok Build
GitHub Copilot CLI
Kiro CLI
```

## GitHub-heavy workflows

Consider:

```text
GitHub Copilot CLI
Codex CLI
```

## Multi-model workflows

Consider:

```text
Cursor Agent
Aider
```

## Local/privacy-oriented experimentation

Consider:

```text
Ollama
+
Aider
+
local model
```

---

# 18. Security Guidance

AI coding agents may be able to:

```text
Read repository files
Modify files
Delete files
Execute shell commands
Run builds
Run tests
Access configured MCP tools
Access environment variables
Interact with Git
```

Therefore:

1. Run agents only inside trusted repositories.
2. Review commands before approving them.
3. Never expose production secrets unnecessarily.
4. Avoid storing API keys directly in source code.
5. Use environment variables or secure credential stores.
6. Review generated diffs before committing.
7. Be especially careful with unattended/headless modes.
8. Limit CI token permissions to what the job actually needs.
9. Use sandboxing/worktrees when available.
10. Do not allow unrestricted shell/write permissions in sensitive environments without review.

---

# 19. API Keys — Recommended Pattern

Use environment variables:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."
```

Do not commit keys into:

```text
Git
README files
shell scripts
Jira tickets
source code
```

For persistent local configuration, prefer the provider's official credential/login mechanism or a secure secret manager.

---

# 20. Useful Verification Commands

```bash
codex --version
claude --version
gemini --version
agy --help
grok --help
copilot --version
agent --version
kiro --help
aider --version
ollama -v
```

---

# 21. Useful `which` Checks

To see where binaries are installed:

```bash
which codex
which claude
which gemini
which agy
which grok
which copilot
which agent
which kiro
which aider
which ollama
```

---

# 22. Fixing `command not found`

For user-local binary installers, ensure:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

For npm global binaries, inspect:

```bash
npm prefix -g
```

and ensure the relevant global binary location is available in `PATH`.

Avoid using `sudo npm install -g ...` as a general workaround for npm permission problems.

A better solution is to use a Node version manager or a user-owned npm prefix.

---

# 23. Recommended Setup for Multiple AI CLIs

Example development workstation:

```text
Ubuntu
│
├── Git
├── Node.js
├── Python
├── Docker
│
├── Codex
├── Claude Code
├── Gemini CLI
├── Antigravity
├── Grok Build
├── GitHub Copilot CLI
├── Cursor Agent
├── Kiro
├── Aider
└── Ollama
```

You do not need to install all of them.

A practical starting set is:

```text
Claude Code
Codex CLI
Gemini / Antigravity
GitHub Copilot CLI
Ollama
```

Add Cursor, Grok, Kiro, or Aider when their workflow/model flexibility is useful.

---

# 24. Example C++ Repository Comparison Test

To compare agents fairly, run the same prompt in each tool.

Prompt:

```text
Do not modify any files.

Analyze this C++/Qt repository.

1. Explain the architecture.
2. Identify the five most important components.
3. Trace the application's startup flow.
4. Identify likely high-risk areas.
5. Identify potential memory/thread/lifetime issues.
6. Cite the relevant source files and functions.
7. Explain any uncertainty instead of guessing.
```

Run from the same clean Git checkout for every agent.

Evaluate:

```text
Accuracy
Source-code understanding
Cross-file reasoning
Hallucination rate
Speed
Context handling
Tool use
Quality of suggested fixes
Token/credit usage
```

---

# 25. AI CLI for Automated Jira Triage

A possible architecture:

```text
Jira Event
    ↓
GitHub Actions
    ↓
AI CLI / Agent
    ↓
Repository context
    +
Jira issue
    +
DLT/log parser
    +
Git history
    ↓
Triage result
    ↓
Jira comment
```

Possible CLI candidates:

```text
Codex CLI
Claude Code
Gemini / Antigravity
GitHub Copilot CLI
Grok Build
Cursor Agent
Kiro
```

For production automation, check each provider's:

```text
Licensing
CI/CD support
Authentication method
Headless mode
Rate limits
Data handling
Enterprise policy
Cost
Model availability
```

before selecting one.

---

# 26. One-Page Command Cheat Sheet

## Codex

```bash
npm install -g @openai/codex
codex
```

## Claude Code

```bash
curl -fsSL claude.ai/install.sh | bash
claude
```

## Gemini CLI

```bash
npm install -g @google/gemini-cli
gemini
```

## Antigravity

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
agy
```

## Grok Build

```bash
curl -fsSL https://x.ai/cli/install.sh | bash
grok
```

## GitHub Copilot CLI

```bash
npm install -g @github/copilot
copilot
```

## Cursor Agent

```bash
curl https://cursor.com/install -fsS | bash
agent
```

## Kiro

```bash
curl -fsSL https://cli.kiro.dev/install | bash
kiro
```

## Aider

```bash
curl -LsSf https://aider.chat/install.sh | sh
aider
```

## Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama
```

Run a model:

```bash
ollama run <model>
```

---

# 27. Official Reference URLs

```text
OpenAI Codex
https://openai.com/codex/

Anthropic Claude Code
https://docs.anthropic.com/

Google Gemini CLI
https://github.com/google-gemini/gemini-cli

Google Antigravity CLI
https://antigravity.google/docs/cli/

xAI Grok Build
https://x.ai/build
https://docs.x.ai/build/

GitHub Copilot CLI
https://docs.github.com/en/copilot/

Cursor CLI
https://cursor.com/cli
https://docs.cursor.com/en/cli/

AWS Kiro CLI
https://kiro.dev/cli/
https://kiro.dev/docs/

Aider
https://aider.chat/docs/

Ollama
https://docs.ollama.com/
```

---

# Final Recommendation

For an Ubuntu software-development workstation, think of the tools in three categories:

```text
PROVIDER-SPECIFIC CODING AGENTS
├── Codex
├── Claude Code
├── Gemini / Antigravity
├── Grok Build
├── GitHub Copilot CLI
└── Kiro

MULTI-MODEL AGENTS
├── Cursor Agent
└── Aider

LOCAL MODEL RUNTIME
└── Ollama
```

For large C++/Qt repository analysis, install two or three agents first and compare them using exactly the same repository and prompts instead of installing every CLI immediately.
