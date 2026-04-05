# sampler.py
import torch
import tiktoken
from gpt2.model import GPT_2, device

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_TYPE   = "gpt2_xl"        # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
MAX_NEW_TOKENS = 200
NUM_SEQUENCES  = 5
TEMPERATURE    = 0.8         # higher = more creative, lower = more focused
TOP_K          = 50          # only sample from top-k most likely tokens

# ── Load ───────────────────────────────────────────────────────────────────────
model = GPT_2.from_pretrained(MODEL_TYPE).to(device)
model.eval()

enc   = tiktoken.get_encoding("gpt2")

# ── Sampler ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    tokens = enc.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # crop to block size if needed
        x_cond  = x[:, -model.config.block_size:]

        logits  = model(x_cond)          # (1, T, vocab)
        logits  = logits[:, -1, :]       # last token only → (1, vocab)
        logits  = logits / temperature

        # top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs   = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
        x        = torch.cat([x, next_tok], dim=1)          # append

    return enc.decode(x[0].tolist())


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = input("Enter prompt (or press Enter for default): ").strip()
    if not prompt:
        prompt = "Once upon a time"

    print(f"\nModel  : {MODEL_TYPE}")
    print(f"Prompt : {prompt}")
    print("-" * 60)

    for i in range(NUM_SEQUENCES):
        print(f"\n--- Sample {i+1} ---")
        print(generate(prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_K))
        print()