import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel




# %%
def get_device():
    """Returns the best available device: CUDA, MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 8552):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU


device=get_device()
set_seed(8552)



# %%
@dataclass
class GPT_Config:
    block_size: int = 1024 # Max Sequence Length
    vocab_size: int = 50257 # number of tokens 50,000 BPE merges +256 bytes tokens + 1 Special Token (<[endoftext]>)
    # For more deteails https://arxiv.org/pdf/2603.02597
    n_layer: int = 12   
    n_head: int = 12    
    n_embd: int = 768   

# %%
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INNIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (345M), n_head=16, hs=64, so nh*hs=C=1024 channels in the Transformer
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        y = self.c_proj(y)
        return y

# %%
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=  nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=   nn.GELU(approximate='none') #note in Orignal GPT-2 Papper they used approx='tanh' 
        #Because TensorFlow back then was slow so they used the approx verison instead, 
        #but Now In modern Days They use approxiamte=none Instead of Using Old Taylor Method
        #For more Details Here is the link https://arxiv.org/pdf/1606.08415
        
        self.c_proj=  nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.GPT_SCALE_INNIT = 1


    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x
    


   

# %%
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)

        self.attn=CausalSelfAttention(config)
        
        self.ln_2=nn.LayerNorm(config.n_embd)
        
        self.mlp=MLP(config)

    def forward(self,x):
      x = x+ self.attn(self.ln_1(x))
      x = x+ self.mlp(self.ln_2(x))
      return x
    
     # Summary of the Flow:
    
     #LayerNorm 1: Clean the data.
     #Attention: Look at other words (context).
     #Add: Merge that context with the original data.
     # LayerNorm 2: Clean the data again.
     # MLP: Think about the context individually.
     # Add: Merge those thoughts with the running "memory."
  


# %%
class GPT_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),#wte=Weight Token Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),#wpe=eight Position Embedding 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INNIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    


    def forward(self,idx):
        
        B,T=idx.size()
        assert T<=self.config.block_size,f"Cannot Forward Sequence of length {T},block size is only {self.config.block_size}"

         
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        pos_emb = self.transformer.wpe(pos) 
        tok_emb = self.transformer.wte(idx) 
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) 
        return logits
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        print("loading weights from pretrained gpt: %s" % model_type)

        
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT_Config(**config_args)
        model = GPT_2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.inference_mode():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.torch.inference_mode():
                    sd[k].copy_(sd_hf[k])

        return model