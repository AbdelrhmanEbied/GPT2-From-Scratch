# GPT-2 From Scratch

A full implementation of GPT-2 built from scratch in PyTorch, including a 
production-grade training loop, data pipeline, and a dockerized REST API.

## What's in this repo

GPT-2/
├── architecture/
│   ├── model.py        # GPT-2 architecture implemented from scratch
│   ├── sampler.py      # Text generation / sampling script
│   └── trainer.py      # Production training loop
└── data/
└── prepare.py      # Downloads, tokenizes and builds DataLoaders


---

## Why the model isn't trained

Training GPT-2 from scratch requires resources that are simply out of reach 
for an individual developer:

- **Data**: ~40GB of text (OpenWebText, 9 billion tokens)
- **Hardware**: Minimum 4x A100 80GB GPUs (~$20/hour on AWS)
- **Time**: 2–3 weeks of continuous training
- **Cost**: Estimated $3,000 - $4,500 in cloud compute
- **OpenAI** spent millions of dollars training the original GPT-2 in 2019

This project focuses on what an individual CAN build — a correct, 
production-quality implementation of the architecture and training 
infrastructure. The architecture has been verified by successfully loading 
official pretrained weights from HuggingFace.

---

## Architecture

Implements the full GPT-2 family:

| Model | Layers | Heads | Embedding | Params |
|---|---|---|---|---|
| GPT-2 Small | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 355M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1.5B |

Key components:
- Causal self-attention with Flash Attention (`F.scaled_dot_product_attention`)
- Pre-norm transformer blocks (LayerNorm before attention and MLP)
- Learned positional embeddings
- Weight tying between token embeddings and output projection

---

## Trainer features

The training loop in `trainer.py` includes:

- Mixed precision training (AMP) with GradScaler
- Gradient accumulation for simulating large batch sizes
- Gradient clipping for training stability
- Multi-GPU support via DistributedDataParallel (DDP)
- CUDA prefetcher for overlapping data transfer with compute
- Learning rate scheduling (per-batch and per-epoch)
- Early stopping with configurable patience
- Automatic checkpointing of best model
- Model EMA support
- WandB / custom logger integration
- `torch.compile()` support

---

## Run the API

The pretrained API loads GPT-2 Medium weights into our custom architecture 
and serves them via FastAPI.

**Requirements**: Docker
```bash
docker pull abdelrhmanebied/gpt2-from-scratch
docker run -p 8000:8000 abdelrhmanebied/gpt2-from-scratch
```

Then send a request:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100}'
```

Or visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Train it yourself

If you have access to a GPU cluster:
```bash
# 1. Install dependencies
pip install torch tiktoken datasets wandb

# 2. Download and tokenize OpenWebText
python data/prepare.py --tokens 100_000_000

# 3. Train
python train.py
```

For full scale training matching the original GPT-2:
```bash
python data/prepare.py --tokens 9_000_000_000  # needs 40GB disk
torchrun --nproc_per_node=8 train.py            # needs 8x A100
```

---

## Load pretrained weights
```python
from model import GPT_2

# Load any GPT-2 variant into our architecture
model = GPT_2.from_pretrained('gpt2')         # 124M
model = GPT_2.from_pretrained('gpt2-medium')  # 355M
model = GPT_2.from_pretrained('gpt2-large')   # 774M
model = GPT_2.from_pretrained('gpt2-xl')      # 1.5B
```

---

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — OpenAI GPT-2 paper
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — Training dataset

---

## Author

**Abdelrhman Ebied**

This is my first transformer implementation and one of my earliest deep learning
projects. I built it to combine everything I was learning at the same time —
PyTorch, distributed training, FastAPI, and Docker — into one end-to-end pipeline.

### What I learned building this

**PyTorch & deep learning**
- How transformers actually work at the code level — attention, residual connections,
  layer norm, positional embeddings — not just theory
- Why training details matter as much as architecture: gradient clipping,
  mixed precision, gradient accumulation, learning rate scheduling
- Common bugs specific to transformers: inplace operations breaking autograd,
  AMP-scheduler desync, DDP early-stopping deadlocks

**Training infrastructure**
- Built a production-grade training loop from scratch with AMP, DDP,
  gradient accumulation, early stopping, and checkpointing
- Learned the hard way about GPU memory management on limited hardware
  (free Colab T4 with 15GB VRAM)
- Used WandB for experiment tracking and monitoring training runs

**Data engineering**
- Tokenizing and processing 100M+ tokens of real web text
- Streaming large datasets without running out of RAM
- Building efficient DataLoaders with CUDA prefetching

**Deployment**
- Serving a PyTorch model via FastAPI
- Containerizing a machine learning application with Docker

### Honest reflection

Since this is my first transformer I definitely didn't get everything right
on the first try. I ran into bugs I'd never seen before, had sessions crash
mid-training, and spent a lot of time understanding why things weren't working.
But that's exactly why I built it — to learn by doing rather than just following
a tutorial.

The architecture is correct and verified against official HuggingFace weights.
The trainer is production-quality. The deployment works.
Training at full scale is the next step — when I get access to real compute.
