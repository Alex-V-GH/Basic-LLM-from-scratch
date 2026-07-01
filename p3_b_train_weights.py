import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import math
import multiprocessing
import time
import re

# ── hiperparámetros ──────────────────────────────────────────────
N_LAYERS    = 8
N_HEADS     = 8
D_MODEL     = 512
D_FF        = 2048
CONTEXT_LEN = 1024
DROPOUT     = 0.1
VOCAB_SIZE  = 32000

LR          = 3e-4
BATCH_SIZE  = 16

TOTAL_TOKENS= 4146018702 # 4,146,018,702    #Poner aca el total arrojado por tokenizador
#crear contador interno real

n_samples   = TOTAL_TOKENS // CONTEXT_LEN          # chunks sin overlap
TOTAL_STEPS = n_samples // BATCH_SIZE
STEPS_DIGITS = len(str(int(TOTAL_STEPS)))
TIMES       = []


# ─────────────────────────────────────────────────────────────────
from live_loss_plot import create_live_loss_plot
def charge_dataloader(file):
    try: 
        with open(file, "r") as f:
            dataloader = [
            (int(line.split(",")[0]),
            float(line.split(",")[1]))
            for line in f.read().splitlines() if line
            ]
    except:
        dataloader = []
    return dataloader 
# ─────────────────────────────────────────────────────────────────
def calc_tiempo(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
# ─────────────────────────────────────────────────────────────────
def check_last_checkpoint(checkpoint):
    last = max(
        (f for f in os.listdir(checkpoint) if f.endswith(".pt")),
        key=lambda f: int(re.search(r'\d+', f).group()),
        default=None
    )
    return os.path.join(checkpoint, last) if last else None
# ─────────────────────────────────────────────────────────────────


class NBModelDataset(Dataset):
    def __init__(self, bin_path, context_len):
        self.bin_path    = bin_path
        self.context_len = context_len
        self.data        = None

        n_bytes     = os.path.getsize(bin_path)
        n_tokens    = n_bytes // 2
        self._len   = (n_tokens - 1) // context_len  # ← sin overlap
        print(f"Total tokens en disco: {n_tokens:,}")
        print(f"Total samples (chunks): {self._len:,}")

    def _open(self):
        if self.data is None:
            self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._open()
        start = idx * self.context_len          # ← stride = context_len
        chunk = self.data[start : start + self.context_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


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

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.drop(self.token_emb(x) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def make_loader(dataset, skip_batches=0):
    #CORREGIR Y HACER QUE USE SKIP BATCHES!
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )


def train(save_every, log_every, checkpoint, token_bin, resume_from, log_file, dataload, model_name, last_chkpt_file, last_chkpt_index):
    last_chkpt_index = last_chkpt_index + save_every
    assert os.path.exists(token_bin), \
        f"No se encontró {token_bin} — corré primero 4a_tokenize_dataset.py"

    os.makedirs(checkpoint, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = NBModelDataset(token_bin, CONTEXT_LEN)

    model     = NewbornModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    steps_per_epoch = len(dataset) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)
    scaler    = torch.cuda.amp.GradScaler()

    # ── resume ────────────────────────────────────────────────────
    start_step  = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Retomando desde {resume_from}...")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step  = ckpt["step"]
        print(f"Retomado en step {start_step}") #12 sec point 1
    print(f"Los parámetros son: \n{model.count_params():,}\n")
    # ─────────────────────────────────────────────────────────────

    print("...............................\nEmpezando loop\n..............................\n")
    step = start_step

    model.train()

    skip_batches = (start_step % steps_per_epoch)
    loader = make_loader(dataset, skip_batches) #around an hour, start 12:10 end 1:30 //11:28 - 

    optimizer.zero_grad()
    start_steps_block = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        step += 1
        if step % log_every == 0:
            elapsed = time.perf_counter() - start_steps_block
            start_steps_block = time.perf_counter()
            TIMES.append(elapsed)

            print(
            f"[{round(100*step/TOTAL_STEPS,2)}%] | step {step:{STEPS_DIGITS}d} / {TOTAL_STEPS}"
            f" | loss {loss.item():.4f} | lr "
            f"{scheduler.get_last_lr()[0]:.2e} | Block Time:{round(elapsed,1)}sec. | "
            f"Until ChkPt: {calc_tiempo(sum(TIMES) / len(TIMES)*(last_chkpt_index + save_every - step)/log_every)}"
            f" | Until End: {calc_tiempo(sum(TIMES) / len(TIMES)*(TOTAL_STEPS-step)/log_every)}")
            
            plot.update(1, step, loss.item())
            dataload.append((step, loss.item()))

        if step % save_every == 0:
            path = os.path.join(checkpoint, f"{model_name}_step{step}.pt")
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
            }, path)
            with open(log_file, "w") as f:
                f.write("\n".join(f"{s},{l}" for s, l in dataload))
            print(f"Checkpoint guardado: {path}")
            last_chkpt_index = last_chkpt_index + save_every

    torch.save(model.state_dict(), r"Models Dev/Rosa/rosa_pretrained.pt")
    print("Preentrenamiento completo.")


if __name__ == "__main__":
    model_name = "Rosa"

    plot = create_live_loss_plot()
    contador = 0
    log = "Models Dev/Rosa/3_b_dataloader.txt"
    dl = charge_dataloader(log)

    save_every = 5_000
    log_every = 10
    checkpoint = r"Models Dev/Rosa/checkpoints"
    token_bin = r"Models Dev/Rosa/rosa_tokens.bin"
    last_chkpt = check_last_checkpoint(checkpoint)
    resume_from = "Models Dev/Rosa/checkpoints/rosa_step"+str(last_chkpt)+".pt"

    for step, loss in dl:
        if contador % 50 == 0:
            plot.update(2,step, loss)
        #este contador es para acelerar en numeros altos de steps tomados.
        contador+=1
    multiprocessing.set_start_method("spawn", force=True)
    train(save_every, log_every, checkpoint, token_bin, resume_from, log, dl, model_name,resume_from,last_chkpt)
    plot.close()  # deja el gráfico visible al final