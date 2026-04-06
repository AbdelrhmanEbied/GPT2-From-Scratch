
from datasets import load_dataset
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_URL    = "Skylion007/openwebtext"
SHARD_PATH  = "openwebtext_tokens.bin"  
T           = 1024                        
BATCH_SIZE  = 128
DTYPE       = np.uint16                  



def build_token_file(path: str):
    print("Tokenising OpenWebText → disk ...")
    enc = tiktoken.get_encoding("gpt2")
    ds  = load_dataset(DATA_URL, split="train", streaming=True)

    
    FLUSH_EVERY = 10_000_000          
    buf   = []
    total = 0

    fp = open(path, "wb")             
    eot = enc.eot_token               
    for i, example in enumerate(ds):
        buf.extend(enc.encode_ordinary(example["text"]))
        buf.append(eot)               

        if len(buf) >= FLUSH_EVERY:
            arr = np.array(buf, dtype=DTYPE)
            arr.tofile(fp)
            total += len(arr)
            buf.clear()
            print(f"  flushed {total:,} tokens so far …", end="\r")

    
    if buf:
        arr = np.array(buf, dtype=DTYPE)
        arr.tofile(fp)
        total += len(arr)

    fp.close()
    print(f"\nDone. Total tokens on disk: {total:,}")


if not os.path.exists(SHARD_PATH):
    build_token_file(SHARD_PATH)
else:
    print(f"Token file already exists: {SHARD_PATH}")



class TokenDataset(Dataset):
    """
    Reads (x, y) pairs straight from the memmap — no copies in RAM.
    x = tokens[i : i+T],  y = tokens[i+1 : i+T+1]
    """
    def __init__(self, path: str, seq_len: int, start: float = 0.0, end: float = 1.0):
        self.data    = np.memmap(path, dtype=DTYPE, mode="r")
        n_seq        = (len(self.data) - 1) // seq_len
        self.seq_len = seq_len
        self.start   = int(start * n_seq)
        self.end     = int(end   * n_seq)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        i   = (self.start + idx) * self.seq_len
        x   = torch.from_numpy(self.data[i     : i + self.seq_len    ].astype(np.int64))
        y   = torch.from_numpy(self.data[i + 1 : i + self.seq_len + 1].astype(np.int64))
        return x, y



train_ds = TokenDataset(SHARD_PATH, T, start=0.0, end=0.8)
val_ds   = TokenDataset(SHARD_PATH, T, start=0.8, end=1.0)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)

print(f"Train sequences : {len(train_ds):,}  |  batches: {len(train_dl):,}")
print(f"Val   sequences : {len(val_ds):,}  |  batches: {len(val_dl):,}")
