# Learning Meta AI / Llama from First Principles

## Goal

This document is a complete learning roadmap for understanding how Meta's Llama models are built, trained, fine-tuned, evaluated, and deployed.

It is designed for someone who wants to go beyond simply *using* Llama and instead understand:

- How a Llama-style model is built from scratch
- How Transformer architecture works
- How tokenization works
- How pretraining works
- How instruction tuning works
- How preference optimization works
- How inference works
- How model weights are stored and loaded
- How distributed training works
- How Llama differs across generations
- How safety and evaluation are added
- How to fine-tune Llama
- How to build a small Llama-like model yourself
- Which parts of Meta's original training pipeline are public and which are not

---

# 1. Important Terminology

## LLM

LLM means:

```text
Large Language Model
```

An LLM is a neural network trained on very large amounts of text, code, and other data.

Its fundamental task during pretraining is usually:

```text
Predict the next token
```

## Foundation Model

A foundation model is a large pretrained model that can later be adapted for many tasks.

Examples:

```text
Llama Base
GPT Base Models
Gemini Foundation Models
Claude Foundation Models
```

## Base Model

A base model is primarily trained using next-token prediction. It is knowledgeable but is not necessarily trained to behave as a conversational assistant.

## Instruction-Tuned Model

An instruction-tuned model is trained to follow requests such as:

```text
Explain this function.
Summarize this document.
Find the bug.
Write a unit test.
```

## Open Weight vs Open Source

Llama is best described as an **open-weight model family**.

Meta provides substantial public materials, including:

```text
Model weights
Reference implementations
Model architecture information
Tokenizer support
Inference code
Fine-tuning examples
Model cards
Safety tooling
Evaluation tooling
Licensing information
```

Meta does not release everything required to reproduce its exact internal training process. Important missing pieces include the complete original training dataset, exact document/URL list, all private datasets, all annotation datasets, all internal quality filters, full training orchestration, every training configuration, all checkpoints, and all proprietary evaluation infrastructure.

---

# 2. High-Level Llama Development Pipeline

```text
Raw Data
   ↓
Cleaning / Filtering / Deduplication
   ↓
Tokenizer
   ↓
Transformer Architecture
   ↓
Random Weight Initialization
   ↓
Pretraining
   ↓
Llama Base
   ↓
Supervised Fine-Tuning
   ↓
Preference Optimization / RL
   ↓
Safety Training
   ↓
Llama Instruct
   ↓
Tools / RAG / Agents / Product Integration
   ↓
Meta AI
```

---

# 3. Mathematics to Learn

## Linear Algebra

Learn vectors, matrices, matrix multiplication, dot products, transpose, tensor shapes, and linear transformations.

## Probability

Learn probability distributions, conditional probability, log probability, softmax, cross entropy, and sampling.

## Calculus

Understand derivatives, partial derivatives, gradients, and the chain rule.

## Optimization

Learn gradient descent, mini-batch gradient descent, Adam, AdamW, learning rate, weight decay, gradient clipping, and schedulers.

---

# 4. Neural Network Fundamentals

Understand:

```text
Neuron
Layer
Activation function
Forward propagation
Loss
Backpropagation
Gradient
Optimizer
Training step
Epoch
Batch
```

Simple conceptual loop:

```python
for batch in training_data:
    prediction = model(batch.inputs)
    loss = loss_function(prediction, batch.targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

# 5. Tokenization

LLMs do not directly process text. They process integer token IDs.

Study:

```text
Token
Vocabulary
Byte Pair Encoding
SentencePiece
Subword tokenization
Special tokens
BOS
EOS
Padding
Chat templates
```

Tokenizer design affects model efficiency, code handling, multilingual support, memory, context length, and training speed.

---

# 6. Transformer Architecture

Study the paper **Attention Is All You Need** and understand:

```text
Embedding
Positional information
Self-attention
Query
Key
Value
Attention score
Multi-head attention
Feed-forward network
Residual connection
Normalization
Decoder-only Transformer
Causal masking
```

Conceptually:

```text
Tokens
   ↓
Embeddings
   ↓
Transformer Block × N
   ↓
Final Normalization
   ↓
Linear Output Projection
   ↓
Vocabulary Logits
   ↓
Sampling
   ↓
Next Token
```

---

# 7. Self-Attention

For an input representation `X`:

```text
Q = XWq
K = XWk
V = XWv
```

Conceptually:

```text
Attention(Q,K,V) = softmax(QKᵀ / sqrt(d)) V
```

Understand why Q/K/V exist, why scaling is needed, why softmax is used, and why causal masks are necessary.

---

# 8. Llama-Specific Architecture Concepts

Study these after basic Transformer architecture:

```text
RMSNorm
RoPE
SwiGLU
GQA
KV Cache
```

Then study modern sparse architectures:

```text
Mixture of Experts
Expert routing
Top-k routing
Shared experts
Active vs total parameters
```

---

# 9. Build a Tiny Transformer Yourself

Before reading billion-parameter implementations, build a tiny model.

Recommended project:

```text
tiny-llama-learning/
```

Implement:

```text
Tokenizer
Embedding
Attention
RoPE
RMSNorm
Feed-forward layer
Transformer block
Language-model head
Training loop
Generation loop
Checkpoint saving/loading
```

Start around:

```text
10M–50M parameters
```

---

# 10. Pretraining

Pretraining creates the base model.

Initially:

```text
Weights = random values
```

The model repeatedly performs:

```text
Forward pass
      ↓
Prediction
      ↓
Cross-entropy loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
Optimizer
      ↓
Updated weights
```

The usual objective is next-token prediction.

---

# 11. Training Data Pipeline

A realistic pipeline generally involves:

```text
Raw documents
      ↓
Format normalization
      ↓
Language detection
      ↓
Quality filtering
      ↓
Safety filtering
      ↓
Deduplication
      ↓
Domain/data weighting
      ↓
Tokenization
      ↓
Sequence packing
      ↓
Training shards
```

Study exact and near-duplicate removal, MinHash, LSH, benchmark contamination, quality scoring, domain balancing, code weighting, multilingual balancing, and synthetic data.

---

# 12. Distributed Training

Learn:

```text
Data Parallelism
Tensor Parallelism
Pipeline Parallelism
Sequence Parallelism
Expert Parallelism
Distributed Data Parallel
Fully Sharded Data Parallel
ZeRO
```

Also study lower-precision training:

```text
FP32
FP16
BF16
TF32
FP8
Mixed precision
Loss scaling
```

---

# 13. Checkpoints

A checkpoint may contain:

```text
Model parameters
Optimizer state
Training step
Scheduler state
Random-number states
Metadata
```

Learn PyTorch checkpointing, Safetensors, sharded checkpoints, distributed checkpointing, and resume training.

---

# 14. Llama Base

After pretraining, the model can complete text, generate code, answer some questions, continue documents, and represent learned knowledge.

But it is not yet guaranteed to behave like a polished assistant.

---

# 15. Supervised Fine-Tuning (SFT)

SFT teaches instruction-following behavior.

```text
Base Model
    +
Instruction Dataset
    ↓
Supervised Fine-Tuning
    ↓
Instruction Model
```

Study instruction datasets, conversation formatting, chat templates, system messages, token masking, assistant-token loss, epochs, learning rate, overfitting, and catastrophic forgetting.

---

# 16. Preference Training

Study:

```text
RLHF
Reward Models
PPO
DPO
Rejection Sampling
RLAIF
GRPO
```

For learning, start with **DPO** before building a complete RLHF pipeline.

---

# 17. Synthetic Data

Modern models increasingly use AI-generated training data.

```text
Teacher Model
     ↓
Generate examples
     ↓
Filter / Verify
     ↓
Train Student Model
```

Study self-instruct, synthetic reasoning data, rejection sampling, teacher/student training, and distillation.

---

# 18. Inference

Inference means using the trained model to generate output.

```text
Prompt
  ↓
Tokenizer
  ↓
Model
  ↓
Logits
  ↓
Sampling
  ↓
Next token
  ↓
Append token
  ↓
Repeat
```

Learn greedy decoding, temperature, top-k, top-p, repetition penalties, stop tokens, and max tokens.

---

# 19. KV Cache

During autoregressive generation, models cache attention keys and values instead of recomputing everything.

Study:

```text
KV-cache memory
Prompt length
Generation length
GQA impact
Paged attention
```

---

# 20. Quantization

Study:

```text
FP16
BF16
INT8
INT4
GPTQ
AWQ
GGUF
bitsandbytes
```

Quantization reduces memory usage and often increases inference speed.

---

# 21. Fine-Tuning Existing Llama Models

Methods:

```text
Full Fine-Tuning
LoRA
QLoRA
Adapters
Prompt tuning
```

Start with **LoRA / QLoRA**.

---

# 22. RAG

RAG means Retrieval-Augmented Generation.

```text
User question
     ↓
Search knowledge base
     ↓
Retrieve relevant content
     ↓
Add content to prompt
     ↓
LLM
     ↓
Answer
```

Use RAG for frequently changing knowledge such as current source code, Jira tickets, logs, commits, architecture documents, and test results.

Use fine-tuning primarily to teach behavior, style, domain patterns, and task-specific decision making.

---

# 23. Tools and Agents

A model becomes much more capable when it can call tools.

```text
LLM
 ├── Search source code
 ├── Parse logs
 ├── Query Jira
 ├── Inspect Git history
 ├── Run tests
 ├── Read build logs
 └── Generate reports
```

Study function calling, tool schemas, MCP, agent loops, planning, tool-result grounding, permissions, and sandboxing.

---

# 24. Safety

Study:

```text
Prompt injection
Jailbreaks
Toxicity
Cybersecurity safety
Content classification
Input filtering
Output filtering
Policy models
Llama Guard
Prompt Guard
CodeShield
```

---

# 25. Evaluation

Study:

```text
Benchmarking
Accuracy
Perplexity
Human evaluation
Pairwise preference
Hallucination tests
Coding benchmarks
Safety benchmarks
Regression testing
Domain-specific evaluation
```

For a C++ engineering model, build your own evaluation set around code explanation, null dereference detection, deadlocks, Git diff review, Qt signal/slot tracing, DLT analysis, Jira severity/component prediction, regression analysis, and root-cause analysis.

---

# 26. Official Meta Llama Repositories

## llama-models

Purpose:

```text
Model definitions
Reference architecture
Tokenizer/model utilities
Model metadata
Model cards
Licenses
Download tooling
Inference examples
```

Repository:

```text
https://github.com/meta-llama/llama-models
```

Clone:

```bash
git clone https://github.com/meta-llama/llama-models.git
```

## llama-cookbook

Purpose:

```text
Fine-tuning
Inference
RAG
Deployment
Application recipes
```

```text
https://github.com/meta-llama/llama-cookbook
```

## PurpleLlama

Purpose:

```text
Safety
Cybersecurity
Llama Guard
Prompt Guard
CodeShield
Safety evaluations
```

```text
https://github.com/meta-llama/PurpleLlama
```

## Original Llama Repository

Useful for studying earlier and simpler implementations.

```text
https://github.com/meta-llama/llama
```

---

# 27. Recommended Source-Code Reading Order

Read in this order:

```text
1. Model configuration
2. Tokenizer
3. Embedding layer
4. RMSNorm
5. RoPE
6. Attention implementation
7. Feed-forward block
8. Transformer block
9. Full Transformer model
10. Output projection
11. Generation code
12. Checkpoint loading
13. Distributed inference
14. Fine-tuning examples
15. Safety tooling
```

For every class/function, answer:

```text
What goes in?
What shape is it?
What happens?
What comes out?
Why is it needed?
```

---

# 28. Learn Tensor Shapes

Track:

```text
B  = batch size
T  = sequence length
D  = hidden dimension
H  = attention heads
Dh = head dimension
```

Typical shapes:

```text
Input tokens          [B, T]
Embeddings            [B, T, D]
Queries               [B, T, H, Dh]
Attention output      [B, T, D]
Vocabulary logits     [B, T, VocabularySize]
```

Understanding shapes makes Transformer code much easier.

---

# 29. Build a Small Llama-Style Model

Recommended progression:

```text
Stage A: Character model               1M–5M parameters
Stage B: Tiny GPT-style Transformer   10M–30M parameters
Stage C: Tiny Llama-style Transformer 30M–150M parameters
Stage D: Pretrain it
Stage E: Instruction-tune it
Stage F: Add preference tuning
Stage G: Add RAG
Stage H: Add tools
```

For the Llama-style stage, add:

```text
RMSNorm
RoPE
SwiGLU
GQA
Llama-style configuration
```

---

# 30. Example Project Architecture

```text
my-llama/
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.py
│
├── model/
│   ├── config.py
│   ├── embedding.py
│   ├── rope.py
│   ├── rmsnorm.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── transformer_block.py
│   └── model.py
│
├── data/
│   ├── download.py
│   ├── clean.py
│   ├── deduplicate.py
│   ├── tokenize.py
│   └── dataset.py
│
├── training/
│   ├── pretrain.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── checkpoint.py
│
├── sft/
│   ├── dataset.py
│   └── train_sft.py
│
├── preference/
│   └── train_dpo.py
│
├── inference/
│   ├── generate.py
│   └── sampling.py
│
├── evaluation/
│   ├── perplexity.py
│   └── benchmarks.py
│
└── README.md
```

---

# 31. Recommended Software Stack

Learn:

```text
Python
PyTorch
CUDA basics
Hugging Face Transformers
Hugging Face Datasets
Tokenizers
SentencePiece
Safetensors
Accelerate
PEFT
TRL
```

Later:

```text
DeepSpeed
FSDP
Megatron-style training
vLLM
TensorRT-LLM
llama.cpp
```

---

# 32. llama.cpp

Study `llama.cpp` to understand efficient local inference, especially if you have a C/C++ background.

Topics:

```text
GGUF
Quantization
CPU inference
GPU offload
KV cache
Sampling
Memory mapping
Optimized kernels
```

---

# 33. vLLM

Study vLLM for production inference.

Topics:

```text
PagedAttention
Batching
KV-cache management
High-throughput serving
OpenAI-compatible APIs
Tensor parallel inference
```

---

# 34. Recommended Llama Generation Study Order

Study broadly in this order:

```text
Llama 2
   ↓
Llama 3
   ↓
Llama 3.1 / 3.x
   ↓
Newer Llama generations
```

Start with dense Llama 3-style architecture before studying more complex MoE and multimodal systems.

---

# 35. What Meta Does Not Publish

Do not expect to find:

```text
A script that exactly recreates Llama
Complete original training corpus
Every data-cleaning rule
All internal model experiments
All human annotation datasets
All proprietary preference data
Exact internal GPU orchestration
All training failures
Every model checkpoint
Internal Meta AI backend source
```

The public ecosystem is sufficient to learn the architecture and build your own model, but not to reproduce Meta's exact model bit-for-bit.

---

# 36. Recommended 12-Stage Learning Order

```text
1. Python + PyTorch
2. Neural-network basics
3. Tokenization
4. Transformer fundamentals
5. Build tiny GPT
6. Learn Llama architecture
7. Build tiny Llama
8. Pretraining
9. Instruction tuning
10. Preference optimization
11. RAG + tools
12. Distributed training + production inference
```

---

# 37. Practical Projects

```text
Project 1: Character-level language model
Project 2: Tiny Transformer trained on text
Project 3: Tiny Llama-style model
Project 4: Fine-tune an existing small Llama model
Project 5: Build RAG over technical documents
Project 6: Build a coding assistant over a C++ repository
Project 7: Add log-analysis tools
Project 8: Add Jira integration
Project 9: Build automated issue triage
Project 10: Benchmark local model vs hosted frontier model
```

---

# 38. Example Specialized Engineering AI

```text
                 Specialized LLM
                       │
          ┌────────────┼────────────┐
          │            │            │
         RAG          Tools       Memory
          │            │            │
     Source code    Git/Jira     History
     Documents      Log parser   Decisions
     Architecture   Tests        Context
          │            │            │
          └────────────┼────────────┘
                       ↓
               Engineering Agent
```

A C++/Qt specialization could be:

```text
Base Llama Model
      ↓
C++/Qt instruction tuning
      ↓
Repository RAG
      ↓
Git tools
      ↓
DLT parser
      ↓
Jira integration
      ↓
Build/test tools
      ↓
Automated triage agent
```

---

# 39. Key Questions You Should Be Able to Answer

By the end, you should be able to explain:

```text
What is a token?
What is an embedding?
What are Q, K and V?
How does self-attention work?
Why is causal masking required?
What is RoPE?
What is RMSNorm?
What is SwiGLU?
What is GQA?
What is MoE?
What is next-token prediction?
How does cross-entropy work?
What does backpropagation change?
What is a checkpoint?
What is SFT?
What is RLHF?
What is DPO?
What is LoRA?
What is QLoRA?
What is RAG?
What is a KV cache?
What is quantization?
How is an LLM distributed across GPUs?
How is a model served efficiently?
What exactly is contained in model weights?
What separates Llama Base from Llama Instruct?
What separates Llama from Meta AI as a product?
```

---

# 40. Recommended Repository Collection

```bash
mkdir -p ~/ai-learning/meta-llama
cd ~/ai-learning/meta-llama

git clone https://github.com/meta-llama/llama-models.git
git clone https://github.com/meta-llama/llama-cookbook.git
git clone https://github.com/meta-llama/PurpleLlama.git
git clone https://github.com/meta-llama/llama.git
```

Suggested structure:

```text
~/ai-learning/meta-llama/
├── llama-models/
├── llama-cookbook/
├── PurpleLlama/
├── llama/
├── notes/
├── experiments/
└── my-llama/
```

---

# 41. Personal Study Notes Structure

```text
notes/
├── 01-neural-networks.md
├── 02-tokenization.md
├── 03-transformers.md
├── 04-attention.md
├── 05-rope.md
├── 06-rmsnorm.md
├── 07-swiglu.md
├── 08-gqa.md
├── 09-pretraining.md
├── 10-distributed-training.md
├── 11-sft.md
├── 12-dpo.md
├── 13-lora.md
├── 14-rag.md
├── 15-inference.md
├── 16-quantization.md
├── 17-safety.md
└── 18-evaluation.md
```

---

# 42. Final Learning Principle

For every important concept:

```text
Read
  ↓
Implement
  ↓
Run
  ↓
Inspect tensors
  ↓
Break it
  ↓
Debug it
  ↓
Measure it
  ↓
Explain it yourself
```

The best way to understand Llama is to build a small version of it.

---

# Summary

```text
Neural Networks
      ↓
Tokenization
      ↓
Transformer
      ↓
Llama Architecture
      ↓
Tiny Llama Implementation
      ↓
Pretraining
      ↓
Distributed Training
      ↓
Llama Base
      ↓
SFT
      ↓
Preference Optimization
      ↓
Llama Instruct
      ↓
Quantization / Inference
      ↓
RAG
      ↓
Tools / MCP
      ↓
Safety
      ↓
Evaluation
      ↓
Production AI Agent
```

This path teaches not only how to use Meta's Llama models, but how the core technology behind modern LLMs is actually built.
