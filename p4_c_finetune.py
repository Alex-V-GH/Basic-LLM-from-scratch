import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import numpy as np
import math
import time

# ── entry points ─────────────────────────────────────────────────
BOS          = "<bos>"
EOS          = "<eos>"
SEP          = "<mask>"
# ─────────────────────────────────────────────────────────────────

# ── hiperparámetros ──────────────────────────────────────────────
N_LAYERS    = 8
N_HEADS     = 8
D_MODEL     = 512
D_FF        = 2048
CONTEXT_LEN = 1024
DROPOUT     = 0.1
VOCAB_SIZE  = 32000

LR          = 3e-5
BATCH_SIZE  = 4
# ─────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_head  = D_MODEL // N_HEADS
        self.qkv  = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        scale = math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) / scale
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_FF), nn.GELU(),
            nn.Linear(D_FF, D_MODEL), nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.attn  = MultiHeadAttention()
        self.ff    = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class NewbornModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(CONTEXT_LEN, D_MODEL)
        self.drop      = nn.Dropout(DROPOUT)
        self.blocks    = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.norm      = nn.LayerNorm(D_MODEL)
        self.head      = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.drop(self.token_emb(x) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class FinetuningDataset(Dataset):
    def __init__(self, bin_path, context_len, sep_token_id):
        self.context_len  = context_len
        self.sep_token_id = sep_token_id

        data     = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n_tokens = len(data)
        self._len = (n_tokens - 1) // context_len

        self.data = data
        print(f"Total tokens: {n_tokens:,} | Samples: {self._len:,}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start = idx * self.context_len
        chunk = self.data[start : start + self.context_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        # máscara: 1 donde es respuesta (después del SEP), 0 donde es input
        mask = torch.zeros(self.context_len, dtype=torch.bool)
        sep_positions = (x == self.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_pos = sep_positions[0].item()
            mask[sep_pos + 1:] = True

        return x, y, mask


def calc_loss(logits, targets, mask, vocab_size):
    logits_flat  = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat    = mask.view(-1)

    loss_full = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")

    # loss solo sobre respuesta
    if mask_flat.any():
        loss_response = loss_full[mask_flat].mean()
    else:
        loss_response = torch.tensor(0.0)

    # loss solo sobre input
    if (~mask_flat).any():
        loss_input = loss_full[~mask_flat].mean()
    else:
        loss_input = torch.tensor(0.0)

    loss_avg = loss_full.mean()

    return loss_response, loss_input, loss_avg


def finetune(root_dir, model_name, epochs, log_every):
    # ── checkpoint a cargar ───────────────────────────────────────────
    CHECKPOINT_PATH = root_dir + model_name + "_pretrained.pt"
    TOKENIZER_PATH  = root_dir + model_name + "_tokenizer.json"
    TOKEN_BIN       = root_dir + "Finetuning_tokens.bin"
    OUTPUT_PT       = root_dir + model_name + "_finetuned.pt"
    LOG_FILE        = root_dir + "finetuning_log.txt"
    # ─────────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    sep_id    = tokenizer.token_to_id(SEP)

    print(f"Device: {device} | SEP token id: {sep_id}")

    dataset = FinetuningDataset(TOKEN_BIN, CONTEXT_LEN, sep_id)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = NewbornModel().to(device)

    # cargar checkpoint
    print(f"Cargando checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Parámetros: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    log_lines = []

    for epoch in range(epochs):
        model.train()
        for step, (x, y, mask) in enumerate(loader):
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x)
                loss_resp, loss_input, loss_avg = calc_loss(logits, y, mask, VOCAB_SIZE)
                loss = loss_resp  # entrenamos sobre respuesta

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if (step + 1) % log_every == 0:
                lr_actual = scheduler.get_last_lr()[0]
                log = (
                    f"epoch {epoch+1:02d} | step {step+1:04d} "
                    f"| LOSS_RESP {loss_resp.item():.4f} "
                    f"| loss_input {loss_input.item():.4f} "
                    f"| loss_avg {loss_avg.item():.4f} "
                    f"| lr {lr_actual:.2e}"
                )
                print(log)
                log_lines.append(log)

    torch.save(model.state_dict(), OUTPUT_PT)
    print(f"\nModelo guardado: {OUTPUT_PT}")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Log guardado: {LOG_FILE}")


if __name__ == "__main__":
    root_dir = r"Models Dev/Rosab/"
    model_name = "Rosa"
    epochs      = 25
    log_every = 1 
    finetune(root_dir, model_name, epochs, log_every)