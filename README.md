# AutoMamba

**Autonomous Mamba SSM research loop** - LLM-guided hyperparameter optimization for Mamba language models.

Implements Karpathy's autoresearch pattern with Claude agent for Mamba/SSM architectures instead of transformers.

## Quick Start (Google Colab)

1. **Open** `AutoMamba_Agent.ipynb` on [Google Colab](https://colab.research.google.com/)
2. **Setup** Runtime → Change runtime type → T4 GPU
3. **Add API key** Secrets → `OPENROUTER_API_KEY` ([get one here](https://openrouter.ai/))
4. **Run** all cells in order

That's it! The agent will autonomously:
- Analyze experiment results
- Suggest hyperparameter changes
- Run training (2-min experiments on T4)
- Keep improvements, discard failures
- Repeat for N experiments

## How It Works

```
┌─────────────────────────────────────────────┐
│  1. Agent reads results + current config    │
│  2. Claude suggests ONE change              │
│  3. Apply change to train.py                │
│  4. Run 2-min training experiment           │
│  5. val_bpb better? → Keep : Revert         │
│  6. Log to results.tsv                      │
│  7. Repeat                                  │
└─────────────────────────────────────────────┘
```

**Metric**: `val_bpb` (bits per byte) - lower is better, vocab-size independent

## Hyperparameters Optimized

- **d_model**: Model dimension (256, 384, 512)
- **n_layer**: Number of Mamba blocks (8, 12, 16, 20)
- **d_state**: SSM state dimension (8, 16, 32, 64) - affects memory capacity
- **d_conv**: Causal conv kernel (2, 3, 4, 8)
- **expand**: Expansion factor (1, 2, 4)
- **lr**: Learning rate (1e-4, 3e-4, 6e-4, 1e-3)
- **batch_size**: (8, 12, 16, 24)
- **seq_len**: Sequence length (512, 1024)

## Files

**Main notebook (upload to Colab):**
- `AutoMamba_Agent.ipynb` - Complete autonomous research loop

**Supporting files (upload when prompted):**
- `prepare_colab.py` - Data download + tokenizer training
- `train_colab.py` - Mamba model implementation

**Documentation:**
- `CLAUDE.md` - Project overview
- `program.md` - Detailed architecture guide

## Example Session

```
EXPERIMENT 1/10
=====================================
🤖 Agent: "Increasing d_state to 32 for richer dynamics"
   OLD: d_state: int = 16
   NEW: d_state: int = 32
🚀 Training...
✅ IMPROVEMENT! 1.1987 (Δ -0.0469)

EXPERIMENT 2/10
=====================================
🤖 Agent: "Trying expand=4 for wider inner dimension"
   OLD: expand: int = 2
   NEW: expand: int = 4
🚀 Training...
❌ No improvement: 1.2103 vs 1.1987 (reverted)

...

🏁 FINAL RESULTS
Best val_bpb: 1.1234
```

## Mamba SSM Architecture

```
Input (B, L, D)
  → LayerNorm
  → Linear: D → 2*D*expand (split into x and gate)
  → x path:
      - Conv1d (causal, kernel=d_conv)
      - SiLU activation
      - SSM scan (selective state space)
        * Δ, B, C projections
        * A matrix (log-space)
        * Cumsum trick for parallel scan
  → Gate: SiLU(z) * ssm_output
  → Linear: D*expand → D
  → Residual
```

**Key advantage**: Linear O(L) complexity vs transformer's O(L²)

## Cost

- **Compute**: Free (Colab T4 GPU, ~12hr runtime)
- **API**: ~$0.50 per 10 experiments (OpenRouter Claude 3.5 Sonnet)
- **Data**: Free (TinyStories, ~400MB download)

## Tips

1. **Keep runtime alive**: Colab disconnects after 90min idle - keep tab open
2. **Start small**: Try 5-10 experiments first to see how it works
3. **Monitor memory**: If OOM errors, reduce `seq_len` or `batch_size`
4. **Extend training**: Change `TIME_BUDGET` in prepare.py for longer experiments (default 120s)
5. **Custom search space**: Modify hyperparameter ranges in Cell 4

## Key Features

✅ **Single notebook** - Everything in one place, runs on free Colab
✅ **LLM agent** - Intelligent exploration, learns from failures
✅ **Fast iteration** - 2-min experiments, ~10-15 per hour
✅ **Keep/revert logic** - Only accumulates improvements
✅ **Automatic logging** - results.tsv tracks all experiments
✅ **Visualization** - Progress plots after completion

## Comparison to Transformers

| Feature | GPT (Transformer) | Mamba (SSM) |
|---------|-------------------|-------------|
| Complexity | O(L²) | O(L) |
| Memory | O(L²) | O(L) |
| Inference | Linear scan | Constant time |
| Key mechanism | Attention | Selective SSM |
| Training speed | Slower | Faster (on long seqs) |

## References

- [Mamba paper](https://arxiv.org/abs/2312.00752) - Gu & Dao, 2023
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) - Original pattern
- [mamba.py](https://github.com/alxndrTL/mamba.py) - Pure PyTorch reference
- [autoresearch-lite](https://github.com/parthwhy/autoresearch-lite) - Colab adaptation

## License

MIT (following autoresearch)

---

**Ready?** Upload `AutoMamba_Agent.ipynb` to Colab and let the agent optimize your Mamba model!
