# AGI system development roadmap

A complete AGI-like system is not just one Transformer. It is a collection of
models, memory systems, tools, learning loops, evaluations, infrastructure, and
safety controls.

Building the projects in this roadmap does not guarantee AGI. The practical goal
is to create an increasingly capable, measurable, and controlled experimental
agent platform.

## Major projects

| Priority | Project | Purpose |
|---:|---|---|
| 1 | `char-completion-transformer` | Learn token prediction and Transformer fundamentals |
| 2 | `tokenizer` | Implement character, word, BPE, and subword tokenization |
| 3 | `instruction-transformer` | Learn prompt/response behavior through supervised fine-tuning |
| 4 | `embedding-model` | Convert text into semantic vectors |
| 5 | `rag` | Retrieve external knowledge before generating answers |
| 6 | `tool-calling-agent` | Let a model use APIs, shell tools, search, and calculators |
| 7 | `memory-system` | Store conversation, episodic, semantic, and user memories |
| 8 | `planner-agent` | Break goals into steps and track execution |
| 9 | `reasoning-model` | Train and evaluate multi-step problem solving |
| 10 | `multimodal-model` | Process text, images, audio, and possibly video |
| 11 | `world-model` | Predict environment states and action consequences |
| 12 | `reinforcement-learning` | Learn actions through rewards and environment interaction |
| 13 | `alignment-training` | Implement instruction tuning, preference learning, RLHF/DPO, and rule-based rewards |
| 14 | `evaluation-system` | Measure accuracy, retrieval, reasoning, safety, cost, and latency |
| 15 | `safety-guardrails` | Enforce permissions, policy checks, sandboxing, monitoring, and human approval |
| 16 | `agent-orchestrator` | Combine models, RAG, tools, memory, planning, and safeguards |

## Target architecture

```text
User
  |
  v
Safety and permission checks
  |
  v
Agent orchestrator
  |-- Planner
  |-- Reasoning/instruction model
  |-- RAG and external knowledge
  |-- Short- and long-term memory
  |-- Tool calling
  `-- World/environment model
  |
  v
Output validation
  |
  v
User
```

## Phase 1: Model fundamentals

Build:

```text
char-completion-transformer
tokenizer
instruction-transformer
embedding-model
```

The current `char-completion-transformer` covers basic autoregressive
generation.

The next important model milestone is `instruction-transformer`. Pretraining
alone produces completion behavior; instruction-following requires demonstrations
and preference/alignment training. OpenAI's published InstructGPT process used
supervised demonstrations, preference comparisons, a reward model, and
reinforcement learning. See
[OpenAI's instruction-following research](https://openai.com/index/instruction-following/).

### Tokenizer project

Implement and compare:

- Character tokenization
- Word tokenization
- Byte-level tokenization
- Byte Pair Encoding (BPE)
- Vocabulary training
- Encode/decode correctness
- Unknown-token behavior
- Compression ratio and token-count evaluation

### Instruction Transformer project

Implement:

- Prompt/response dataset format
- Supervised fine-tuning
- Conversation templates
- Attention masking for prompt and response tokens
- Instruction-following evaluation
- Checkpoint and dataset provenance

### Embedding model project

Implement:

- Positive and negative text pairs
- Contrastive training
- Vector normalization
- Similarity search
- Retrieval-quality evaluation

## Phase 2: Knowledge systems

Build:

```text
document-ingestion
embedding-model
vector-store
rag
rag-evaluation
```

RAG combines parametric model knowledge with an external retrievable index. It
allows knowledge to be updated without retraining the base model. See the
[original Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401).

A RAG project should contain:

```text
rag/
|-- ingest
|-- chunk
|-- embed
|-- index
|-- retrieve
|-- rerank
|-- generate
`-- evaluate
```

Important RAG evaluations include:

- Retrieval recall and precision
- Chunking quality
- Reranking quality
- Answer faithfulness
- Citation correctness
- Unsupported-claim detection
- Retrieval and generation latency

## Phase 3: Agent capabilities

Build:

```text
tool-calling-agent
memory-system
planner-agent
reflection-agent
agent-orchestrator
```

Use a controlled execution path:

```text
Model proposes action
        |
        v
Permission checker
        |
        v
Sandboxed tool execution
        |
        v
Result validation
        |
        v
Model receives result
```

### Tool-calling agent

Implement:

- Typed tool schemas
- Argument validation
- Timeouts and cancellation
- Permission checks
- Sandboxed execution
- Result-size limits
- Audit logs
- Human approval for consequential actions

### Memory system

Separate:

- Working memory: current task context
- Conversation memory: previous messages
- Episodic memory: previous tasks and outcomes
- Semantic memory: facts and concepts
- User memory: explicitly permitted preferences

Memory retrieval must be evaluated for relevance, privacy, deletion, poisoning,
and stale information.

### Planner and orchestrator

Implement:

- Goal decomposition
- Dependency-aware steps
- State and progress tracking
- Tool selection
- Failure recovery
- Stop conditions
- Budget limits
- Human escalation

## Phase 4: Learning and reasoning

Build:

```text
supervised-fine-tuning
preference-dataset
reward-model
dpo-or-rlhf
reasoning-evaluation
reinforcement-learning
world-model
```

Human preference learning matters because simple automatic objectives often fail
to represent the behavior people actually want. See
[OpenAI's human-preference research](https://openai.com/index/learning-from-human-preferences/).

### Reasoning model

Develop:

- Verifiable math and code tasks
- Search and planning environments
- Process and outcome evaluation
- Self-consistency experiments
- Tool-assisted reasoning
- Tests resistant to memorization and contamination

Do not treat longer generated explanations as proof of better reasoning. Measure
task correctness, robustness, calibration, and generalization.

### Reinforcement learning and world models

Implement first in controlled environments:

- Explicit observations and actions
- Deterministic simulators where possible
- Clear reward definitions
- Reward-hacking tests
- Offline evaluation
- Strict resource and action limits

## Phase 5: Safety, evaluation, and operations

Build:

```text
evaluation-system
red-team-suite
safety-guardrails
permission-system
observability
model-registry
dataset-registry
experiment-tracker
```

Safety should be present throughout development, not added only at the end.
Frontier development combines data filtering, behavioral evaluations, expert
red-teaming, model-level alignment, deployment monitoring, and enforcement. See
[OpenAI's GPT-4 safety approach](https://openai.com/index/gpt-4-research/).

### Evaluation system

Track:

- Model and dataset version
- Reproducible configuration
- Held-out test performance
- Instruction following
- Retrieval and citation quality
- Hallucination and calibration
- Tool-use correctness
- Prompt injection resistance
- Data leakage and memorization
- Harmful capability and safety behavior
- Latency, memory, compute, and cost

### Safety guardrails

Implement:

- Least-privilege permissions
- Sandboxed tools
- Input and output policy checks
- Prompt-injection defenses
- Secrets isolation
- Rate and resource limits
- Human approval boundaries
- Immutable audit records
- Emergency stop mechanisms

Rule-based rewards can complement human feedback for behaviors that can be
described using clear policies. See
[OpenAI's rule-based rewards research](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/).

## Proposed repository structure

```text
ai-model-dev/
|-- .devcontainer/
|-- ai-dev-container.sh
|-- bin/
|   |-- char-completion-transformer
|   |-- tokenizer
|   |-- instruction-transformer
|   |-- embedding-model
|   |-- rag
|   |-- memory-system
|   |-- tool-agent
|   `-- evaluator
|-- models/
|-- datasets/
|-- cache/
|-- char-completion-transformer/
|-- tokenizer/
|-- instruction-transformer/
|-- embedding-model/
|-- vector-store/
|-- rag/
|-- memory-system/
|-- tool-calling-agent/
|-- planner-agent/
|-- reasoning-model/
|-- reinforcement-learning/
|-- evaluation-system/
|-- safety-guardrails/
`-- agent-orchestrator/
```

Each project command should follow the shared interface where applicable:

```bash
project-name build
project-name train
project-name prompt
project-name test
project-name status
project-name help
```

## Recommended implementation order

1. Complete and evaluate `char-completion-transformer`.
2. Build `tokenizer`.
3. Build `instruction-transformer`.
4. Build `embedding-model`.
5. Build `vector-store` and `rag`.
6. Build `evaluation-system` alongside every preceding project.
7. Add `tool-calling-agent` with strict permissions and sandboxing.
8. Add `memory-system`.
9. Add `planner-agent`.
10. Add alignment and preference-learning experiments.
11. Add controlled reinforcement-learning and world-model experiments.
12. Connect mature components through `agent-orchestrator`.

The immediate next project should be `tokenizer`, followed by
`instruction-transformer`, and then `embedding-model` plus `rag`. This
sequence builds naturally on the current project without prematurely attempting
an infeasible frontier-scale model.

