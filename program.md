# automamba

This is an experiment to have the LLM optimize Mamba SSM architectures.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context (if exists).
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (or Mac MPS). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Architecture: Mamba SSM

Unlike the Transformer's quadratic attention, Mamba uses a **Selective State Space Model**:

```
Input → Embedding → [Mamba Block] × N → RMSNorm → LM Head
```

Each Mamba Block implements:
1. **Pre-norm** (RMSNorm on input)
2. **Linear projection**: d_model → 2*d_inner (split into x path and residual)
3. **Causal conv1d** (kernel=d_conv, groups=d_inner) - captures local context
4. **SiLU activation**
5. **SSM scan**: learns state transition dynamics
   - **A** (state transition matrix): fixed negative reals for stability
   - **B, C** (input/output matrices): **input-dependent** (the "selective" part)
   - **Δ (dt, step size)**: **input-dependent**, controls discretization
   - **D** (skip connection): direct input→output path
6. **Gating**: SiLU(residual) * ssm_output
7. **Output projection**: d_inner → d_model
8. **Residual connection**: output + input

**Key difference from Transformer:** O(L) instead of O(L²) complexity, constant inference speed regardless of sequence length.

**Current implementation**: The `selective_scan()` method uses a **sequential loop** over time steps. This is simple and correct but slow. One major optimization opportunity is to replace this with a **cumsum trick** (vectorized in PyTorch) for 3-10x speedup. See mamba-tiny reference for implementation.

## Hyperparameters You Can Modify

### Core Architecture (in MambaConfig)

- **`d_model`**: 256, 384, 512, 768 (embedding dimension)
  - Higher = more capacity but slower
  - Target: ~30-40M total params

- **`n_layer`**: 8, 12, 16, 20, 24 (number of Mamba blocks)
  - Depth vs width tradeoff
  - More layers = longer compile time

- **`d_state`**: 8, 16, 32, 64 (SSM state dimension)
  - Higher = richer state dynamics but slower
  - Major impact on model expressiveness

- **`d_conv`**: 2, 3, 4, 8 (causal conv kernel size)
  - Local context window before SSM
  - Usually 4 works well

- **`expand`**: 1, 2, 4 (inner dimension multiplier)
  - d_inner = d_model * expand
  - Higher = more parameters in SSM blocks

- **`dt_rank`**: auto (ceil(d_model/16)), or manual
  - Rank of delta (dt) projection
  - Usually auto is fine

### Training (in train() function)

- **`batch_size`**: 4, 8, 16, 32
  - Mac MPS may OOM above 16
  - Smaller batch = more steps in 5 min = better learning

- **`learning_rate`**: 1e-4 to 1e-3
  - SSMs can be sensitive to LR
  - Try 3e-4, 5e-4, 1e-3

- **`weight_decay`**: 0.0, 0.01, 0.1
  - Regularization strength

- **`optimizer`**: AdamW, Adam, Lion, Muon
  - Current: AdamW with betas=(0.9, 0.95)
  - Can try different optimizers or betas

- **`grad_clip`**: 0.5, 1.0, 5.0
  - Gradient clipping threshold
  - Helps with training stability

### Advanced Modifications

**A initialization** (in MambaBlock.__init__):
- Current: `log(arange(1, d_state+1))`
- Try: HiPPO initialization, random negative reals, learned scaling

**SSM scan algorithm** (in MambaBlock.selective_scan):
- Current: sequential loop (simple but slow)
- **Major speedup**: Replace with cumsum trick from mamba-tiny
- This alone could give 3-10x more training steps in 5 minutes

**Δ (dt) dynamics** (in MambaBlock.ssm):
- Current: softplus(linear_proj(x))
- Try: different activation ranges, learned bias, temperature scaling

**Hybrid architectures**:
- Mix 1-2 attention layers among Mamba layers (Jamba-style)
- Add small MLP blocks after some Mamba blocks

**Normalization**:
- Current: RMSNorm
- Try: LayerNorm, different eps values, pre-norm vs post-norm

**Mixed precision**:
- Current: all float32 on MPS
- On CUDA: can try bfloat16 for non-SSM parts

## Known Issues

**Numerical stability:** SSM scan is sensitive to dtype. The current implementation keeps A, Δ discretization, and scan operations in float32 even if the rest of the model uses bfloat16. Do not change this without careful testing.

**Mac MPS quirks:**
- No torch.compile support (MPS doesn't support it)
- May OOM with large batches (reduce to 8 or 4 if needed)
- No native bfloat16 autocast (stay in float32 on MPS)
- Slower than CUDA but sufficient for 5-min experiments

**Sequential scan bottleneck:** The biggest performance bottleneck is the for-loop in `selective_scan()`. Replacing this with the cumsum trick would be a major win and is a prime target for optimization.

**If loss is NaN:** Reduce learning rate or increase gradient clipping. SSMs can be sensitive to large updates.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
training_seconds: 300.1
num_steps:        953
num_params_M:     35.2
depth:            12
```

Extract the key metric:
```bash
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3) — use 0.0 for crashes (MPS doesn't report memory)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	0.0	keep	baseline mamba config
b2c3d4e	1.201234	0.0	keep	d_state 16 → 32
c3d4e5f	1.256789	0.0	discard	d_conv 4 → 8 (worse)
d4e5f6g	0.000000	0.0	crash	batch_size 64 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the code for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Baseline Target

Baseline with default config (d_model=384, n_layer=12, d_state=16, expand=2) should achieve:
- val_bpb: ~1.2-1.5
- params: ~25-35M
- steps: ~100-200 in 5 minutes

Goals after optimization:
- val_bpb < 1.0: Good
- val_bpb < 0.95: Excellent
- val_bpb < 0.90: Outstanding

## High-Priority Optimization Opportunities

1. **SSM scan algorithm** (sequential → cumsum trick): 3-10x speed = 3-10x more steps = major val_bpb improvement
2. **d_state tuning** (16 → 32 or 64): richer dynamics
3. **Learning rate** (3e-4 → 5e-4 or 1e-3): faster convergence
4. **expand** (2 → 3 or 4): more capacity
5. **Batch size** (if memory allows): better gradient estimates
6. **Optimizer** (AdamW → alternatives): may train faster

Good luck optimizing Mamba!
