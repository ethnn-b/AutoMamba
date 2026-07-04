# AutoMamba

Autoresearch-style autonomous experiment loop for Mamba/SSM architectures. Fork Karpathy's autoresearch pattern but replace the GPT transformer with a Mamba selective state space model. Same data, same metric, different architecture class.

## Goal

Build a single-GPU, single-file Mamba training setup that an LLM agent can autonomously modify and improve overnight — exactly like autoresearch does for GPT, but exploring the SSM design space instead.

## Architecture

### Reference repos (use these as source material, do NOT clone them wholesale)
- **autoresearch**: https://github.com/karpathy/autoresearch — the pattern we're replicating
- **mamba.py**: https://github.com/alxndrTL/mamba.py — pure PyTorch Mamba with training, bfloat16, muP
- **mamba-minimal**: https://github.com/johnma2006/mamba-minimal — ~300 line single-file reference
- **mamba-tiny**: https://github.com/PeaBrane/mamba-tiny — cumsum trick, no custom CUDA

### What to build (3 files)

1. **`prepare.py`** — Data prep + eval utilities (fixed, agent cannot touch)
   - Download TinyStories dataset (same as autoresearch)
   - Train BPE tokenizer (same as autoresearch)
   - Dataloader and val_bpb evaluation function
   - Copy/adapt directly from autoresearch's prepare.py

2. **`train.py`** — Mamba model + training loop (the file the agent edits)
   - Single file, ~400-600 lines
   - Full Mamba model definition (no external mamba-ssm dependency)
   - Pure PyTorch — no custom CUDA kernels (agent can't compile those)
   - Training loop with 5-minute wall-clock budget
   - Logs val_bpb at end
   - Must be self-contained: model + optimizer + loop all in one file

3. **`program.md`** — Agent instructions (the file the human edits)
   - Describes the Mamba architecture and what knobs exist
   - Constraints (don't break prepare.py interface, don't exceed 5 min, etc.)
   - Success metric: lowest val_bpb

### Mamba block structure to implement

```
Input (B, L, D)
  → LayerNorm / RMSNorm
  → Linear projection: D → D*expand*2 (split into x path and z gate)
  → x path:
      → 1D causal conv (kernel=d_conv, groups=d_inner)
      → SiLU activation
      → SSM scan:
          - Project x → Δ (dt), B, C  via x_proj linear
          - A stored in log-space, shape (d_inner, d_state)
          - D skip connection, shape (d_inner,)
          - Parallel associative scan OR cumsum trick (no custom CUDA)
          - Output shape: (B, L, d_inner)
  → z gate: SiLU(z) * ssm_output
  → Linear projection: D*expand → D
  → Residual connection
  → Output (B, L, D)
```

### Target model size: ~25-50M params

```
d_model:    384 or 512
n_layers:   12-24
d_state:    16
d_conv:     4
expand:     2
vocab_size: from tokenizer (likely 4096-8192 for TinyStories)
max_seq_len: 512 or 1024
```

### SSM scan — use the cumsum trick (pure PyTorch, no Triton)

The parallel associative scan requires custom kernels. Instead, use the log-cumsum-exp trick from mamba-tiny:
- Express the recurrence as a ratio of two cumulative sums
- Fully vectorized in PyTorch, no for-loops over time steps
- Slower than Triton scan but fast enough for small models on one GPU

### Optimizer

Start with AdamW. Muon is tuned for transformers and may not transfer well to SSMs. Let the agent discover what works — that's the point.

### Known gotcha: fp32 sensitivity

SSMs are sensitive to recurrent dynamics. The SSM scan (A, Δ discretization, cumulative products) should stay in fp32 even if the rest of the model uses bfloat16. Implement mixed precision carefully:
- Embeddings, projections, norms: bfloat16 fine
- SSM scan internals: keep in fp32
- If the agent discovers instabilities, this is likely why

## Build order

1. Get prepare.py working first — download data, tokenize, verify dataloader
2. Implement the Mamba model in train.py — test forward pass shapes
3. Wire up training loop with 5-min budget and val_bpb logging
4. Verify one full 5-min run produces reasonable loss curve (should see loss decrease)
5. Write program.md with Mamba-specific agent instructions
6. Test one agent loop manually (run agent, let it propose a change, train, check val_bpb)

## Design space for the agent to explore (document in program.md)

These are the knobs the agent should know about:
- `d_state`: SSM state dimension (8, 16, 32, 64) — richer dynamics vs memory
- `d_conv`: causal conv kernel (2, 3, 4, 8) — local context before SSM
- `expand`: inner dim multiplier (1, 2, 4) — width vs compute
- A initialization: log-space real, HiPPO, random, learned complex
- Δ (dt) initialization: uniform, learned bias, input-dependent scaling
- Normalization: LayerNorm vs RMSNorm, pre-norm vs post-norm
- Gating: SiLU vs GELU vs Swish
- Residual scaling: standard vs scaled residuals
- Hybrid blocks: mix in 1-2 attention layers among Mamba layers (Jamba-style)
- Optimizer: AdamW vs Muon vs Lion vs schedule variations
- LR schedule: cosine, linear warmup length, peak LR
- Sequence length and batch size tradeoffs within the 5-min budget

## Constraints

- Python 3.10+, PyTorch only (+ numpy, tokenizers for prepare.py)
- No mamba-ssm pip package — we want the model fully in train.py so the agent can modify anything
- No Triton or custom CUDA — pure PyTorch ops only
- Single NVIDIA GPU (should work on 3090/4090/A100/H100)
- All experiments run for exactly 5 minutes wall clock (excluding compilation/warmup)
- val_bpb is the only metric — lower is better, vocab-size-independent
