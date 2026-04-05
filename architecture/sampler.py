import torch
import tiktoken
import argparse
from gpt2.model import GPT_2, device

@torch.inference_mode()
def generate(model, enc, prompt, max_new_tokens, temperature, top_k):
    tokens = enc.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.block_size:]
        logits  = model(x_cond)[:, -1, :] / temperature

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        x        = torch.cat([x, next_tok], dim=1)

    return enc.decode(x[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--prompt",      type=str,   default=None)
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=50)
    parser.add_argument("--num_samples", type=int,   default=5)
    args = parser.parse_args()

    model = GPT_2.from_pretrained(args.model).to(device)
    model.eval()
    enc   = tiktoken.get_encoding("gpt2")

    prompt = args.prompt or input("Enter prompt (or press Enter for default): ").strip() or "Once upon a time"

    print(f"\nModel      : {args.model}")
    print(f"Prompt     : {prompt}")
    print(f"Temperature: {args.temperature} | Top-k: {args.top_k}")
    print("-" * 60)

    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1} ---")
        print(generate(model, enc, prompt, args.max_tokens, args.temperature, args.top_k))