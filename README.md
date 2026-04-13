#  Decoder-Only Transformer Internals
### *A complete end-to-end walkthrough using `microsoft/Phi-3-mini-4k-instruct`*


## About

This notebook demystifies how a **decoder-only Transformer language model** actually works — from raw text all the way to generated output — using nothing but PyTorch and Hugging Face Transformers.

Instead of treating the model as a black box, we crack it open step by step:

```
"The capital of France is"
        │
        ▼  Tokenizer
input_ids  [batch, seq_len]
        │
        ▼  Embedding + 32 Transformer Blocks
hidden_states  [batch, seq_len, 3072]
        │
        ▼  LM Head (Linear projection)
logits  [batch, seq_len, 32064]
        │
        ▼  Decoding Strategy
next_token_id
        │
        ▼  Repeat until EOS
"Paris, the capital of France."
```


---

## What's Inside

| Cell | Topic | Key concepts |
|------|-------|-------------|
| 1 | Install Dependencies | `transformers`, `torch`, `accelerate`, version pinning |
| 2 | Import Libraries | Device detection, GPU/CPU setup |
| 3 | Load Model & Tokenizer | `AutoModelForCausalLM`, `trust_remote_code`, fp16, `device_map` |
| 4 | Tokenization | `input_ids`, `attention_mask`, PyTorch tensor shapes |
| 5 | Manual Forward Pass | Transformer body, hidden states, `[batch, seq, d_model]` |
| 6 | LM Head | Logits, softmax, top-k predictions with probabilities |
| 7 | Greedy Decoding | `argmax`, autoregressive loop, step-by-step token generation |
| 8 | Pipeline API | `pipeline()` vs manual inference — pros, cons, pseudocode |
| 9 | Decoding Strategies | Greedy, sampling, temperature, top-K, top-P (nucleus) |
| 10 | Mini Chat Demo | Multi-turn chat with conversation history management |
| Bonus | Architecture | Full model structure, hyperparameters, GQA explained |


## Concepts You'll Learn

### Tokenization
How text becomes integers that a neural network can process, using Byte-Pair Encoding (BPE).

### PyTorch Tensors
What tensor shapes like `[batch_size, seq_len, hidden_dim]` mean and why they matter.

### Transformer Architecture
- **Embedding layer** — token IDs → dense vectors
- **Rotary Positional Encoding (RoPE)** — how position is injected into attention
- **Grouped-Query Attention (GQA)** — Phi-3's memory-efficient attention variant
- **SwiGLU MLP** — the feed-forward sublayer
- **Residual connections & RMSNorm** — training stability tricks

### Decoding Strategies

| Strategy | Setting | Best for |
|----------|---------|----------|
| Greedy | `do_sample=False` | Factual, deterministic tasks |
| Temperature | `temperature=0.7` | Balancing creativity vs focus |
| Top-K | `top_k=50` | Preventing low-prob token accidents |
| Top-P (nucleus) | `top_p=0.9` | Production chatbots |
| Combined | `top_p=0.9, temperature=0.7` | Most real-world applications |

---

