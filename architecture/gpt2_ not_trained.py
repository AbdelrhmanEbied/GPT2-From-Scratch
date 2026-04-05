from gpt2.model import GPT_2, GPT_Config

def get_gpt2_config(model_type):
    configs = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")

    return GPT_Config(
        block_size=1024,
        vocab_size=50257,
        **configs[model_type]
    )

# --- Test them all ---
model_sizes = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

for size in model_sizes:
    config = get_gpt2_config(size)
    model = GPT_2(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    # Note: GPT-2 uses Weight Tying (wte == lm_head), so we subtract one instance 
    # if you want to match the "official" parameter counts exactly.
    tied_params = total_params - model.lm_head.weight.numel()
    
    print(f"{size:12} | Layers: {config.n_layer} | Embed: {config.n_embd} | Params: {tied_params:,}")
