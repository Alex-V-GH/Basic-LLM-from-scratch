import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import multiprocessing
import time
import re
from tokenizers import Tokenizer # for p4c
from torch.utils.data import Subset

# ── entry points ─────────────────────────────────────────────────
BOS          = "<bos>"
EOS          = "<eos>"
SEP          = "<mask>"
# ── hiperparámetros ──────────────────────────────────────────────
N_LAYERS    = 8
N_HEADS     = 8
D_MODEL     = 512
D_FF        = 2048
CONTEXT_LEN = 1024
DROPOUT     = 0.1
VOCAB_SIZE  = 32000

LR          = 3e-4
BATCH_SIZE  = 8

FT_LR          = 3e-3
FT_BATCH_SIZE  = 4

def calc_sum_valiws(total_tokens, ft = False):
    global TOTAL_TOKENS
    global TOTAL_STEPS
    global STEPS_DIGITS
    global TIMES

    TOTAL_TOKENS = total_tokens
    n_samples   = TOTAL_TOKENS // CONTEXT_LEN          # chunks sin overlap
    TOTAL_STEPS = n_samples // BATCH_SIZE
    if ft:
        TOTAL_STEPS = n_samples // FT_BATCH_SIZE
    STEPS_DIGITS = len(str(int(TOTAL_STEPS)))
    TIMES       = []
# ─────────────────────────────────────────────────────────────────
from live_loss_plot import create_live_loss_plot
def charge_dataloader(file):#Llamar distinto post LLP upgrade
    try: 
        with open(file, "r") as f:
            dataloader = [
            (int(line.split(",")[0]),
            float(line.split(",")[1]),
            float(line.split(",")[2]),
            float(line.split(",")[3]))
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
def check_last_checkpoint_file(checkpoint, model_name):
    last = max(
        (f for f in os.listdir(checkpoint) if f.endswith(".pt")),
        key=lambda f: int(re.search(r'\d+', f).group()),
        default=None
    )
    if last:
        # "Rosa_step5000.pt" → sacar "Rosa_step" del inicio → "5000.pt"
        # → sacar ".pt" del final → "5000"
        numero_str = last.removeprefix(f"{model_name}_step").removesuffix(".pt")
        last_index = int(numero_str)
    else:
        last_index = 0

    return os.path.join(checkpoint, last) if last else None, last_index
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


def calc_ft_loss(logits, targets, mask, vocab_size):
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


def make_loader(dataset, btch_sz, skip_batches=0, shufflee = True):
    skip_samples = skip_batches * btch_sz
    if skip_samples > 0:
        indices = range(skip_samples, len(dataset))
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=btch_sz,
        shuffle=shufflee,#COMENTAR SI TIRA VALUE:ERROR
        pin_memory=True,
        num_workers=4,
    )

def train_loop(loader, device, model, scaler, optimizer, scheduler, log_every, plot, dataload, save_every, checkpoint_dir, log_file, i, x, y, mask,epoch, epochs,finetune,step,last_chkpt_index,model_name):
    global start_steps_block
    x, y = x.to(device), y.to(device)
    if mask is not None: mask = mask.to(device)
    with torch.cuda.amp.autocast():
        logits, loss = model(x, y)
        loss_resp, loss_input, loss_avg = calc_ft_loss(logits, y, mask, VOCAB_SIZE)
        if mask is not None: loss = loss_resp #si hay mask, es FT. Si es ft, la loss se calcula solo sobre la respuesta.

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

    step += 1
    if step % log_every == 0:
        learn_rate = scheduler.get_last_lr()[0]
        if mask is not None: learn_rate = scheduler.get_last_lr()[0] #solo para ft
        elapsed = time.perf_counter() - start_steps_block
        start_steps_block = time.perf_counter()
        TIMES.append(elapsed)
        loss_relevante = loss.item()
        if finetune: loss_relevante = loss_resp.item() 

        print(f"[EPOCH {epoch+1:02d}/{epochs:02d}] "
        f"[{round(100*step/TOTAL_STEPS,2)}%] | step {step:{STEPS_DIGITS}d} / {TOTAL_STEPS} "
        f"| loss {loss_relevante:.4f} | lr {learn_rate:.6e} "
        f"| Block Time:{round(elapsed,1)}sec. "
        f"| loss_input {loss_input.item():.4f} "
        f"| loss_avg {loss_avg.item():.4f} "
        f"| Until ChkPt: {calc_tiempo(sum(TIMES) / len(TIMES)*(last_chkpt_index + save_every - step)/log_every)} "
        f"| Until End: {calc_tiempo(sum(TIMES) / len(TIMES)*(TOTAL_STEPS-step)/log_every)}|")

        plot.update(4, step+epoch*TOTAL_STEPS, loss_relevante, "Current training loss", ("#B40000"),refresh=(step % 2*log_every == 0))
        plot.update(5, step+epoch*TOTAL_STEPS, learn_rate, "Current lr", ("#0000B4"),refresh=(step % 2*log_every == 0))
        plot.update(6, step+epoch*TOTAL_STEPS, elapsed, "Current times per block", ("#00B400"),refresh=(step % 2*log_every == 0))

        dataload.append((step+epoch*TOTAL_STEPS, loss_relevante, learn_rate, elapsed))

    if step+epoch*TOTAL_STEPS % save_every == 0:
        path = os.path.join(checkpoint_dir, f"{model_name}_step{step}.pt")
        torch.save({
            "step":      step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
        }, path)
        with open(log_file, "w") as f:
            f.write("\n".join(f"{s},{l},{lr},{t}" for s, l, lr, t in dataload))
        print(f"Checkpoint guardado: {path}")
        last_chkpt_index = last_chkpt_index + save_every
    return start_steps_block


def train(save_every : int, log_every : int, checkpoint_dir, token_bin, log_file,
          dataload, model_name, last_chkpt_file, last_chkpt_index,plot,root_dir,epochs = 25,finetune =False):
    tokenizer_path  = root_dir + model_name + "_tokenizer.json"
    last_chkpt_index = last_chkpt_index + save_every
    assert os.path.exists(token_bin), \
        f"No se encontró {token_bin} — corré primero 4a_tokenize_dataset.py"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    sep_id    = tokenizer.token_to_id(SEP)

    model     = NewbornModel().to(device)

    l_rate = LR
    btch_sz = BATCH_SIZE
    if finetune:
        l_rate = FT_LR
        btch_sz = FT_BATCH_SIZE
    optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, weight_decay=0.1)


    # ── resume 
    print (f"last chkpt es      {last_chkpt_file}")
    if last_chkpt_file:
        ckpt = torch.load(last_chkpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Cargando checkpoint: {last_chkpt_file}")
        print(f"Parámetros: {model.count_params():,}")
    # ──────────────────────────────────────────────────────────────
    model.train()

    scaler    = torch.cuda.amp.GradScaler()
    start_step  = 0
    step = 0
    if finetune:
        dataset = FinetuningDataset(token_bin, CONTEXT_LEN, sep_id)
        steps_per_epoch = len(dataset) // btch_sz
        skip_batches = (start_step % steps_per_epoch)
        loader  = make_loader(dataset,btch_sz,skip_batches)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader) * epochs, eta_min=1e-4)
        output_final = root_dir+model_name+"_Finetuned.pt"
    else:
        dataset = NBModelDataset(token_bin, CONTEXT_LEN)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)
        output_final = root_dir+model_name+"_pretrained.pt"
        # ── resume ────────────────────────────────────────────────────
        

    if last_chkpt_file and os.path.exists(last_chkpt_file):
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step  = ckpt["step"]
        print(f"Retomado en step {start_step}")
        step = start_step
        
        steps_per_epoch = len(dataset) // btch_sz
        skip_batches = (start_step % steps_per_epoch)
        loader = make_loader(dataset,btch_sz, skip_batches,False) #around an hour, start 12:10 end 1:30 //11:28 - 
        #train_loop
    print("...............................\nEmpezando loop\n...............................\n")
    global start_steps_block
    for epoch in range(epochs):
        beginning = True
        for i, batch in enumerate(loader):
            if i < start_step: 
                print ("pass")
                print(i)
            else:          
                x, y, *rest = batch
                mask = rest[0] if finetune else False
                if beginning: 
                    start_steps_block = time.perf_counter()
                    beginning = False
                train_loop(loader, device, model, scaler, optimizer, scheduler, log_every, plot, dataload, save_every, checkpoint_dir, log_file, i, x, y, mask, epoch, epochs,finetune,i,last_chkpt_index,model_name)
#----------------------------------------------------save final
    torch.save({"step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                },output_final)
    print(f"Preentrenamiento completo.\nModelo guardado: {output_final}")




def train_wrapper(root_dir,model_name,finetune = False,total_tokens = 77425724, epochs = 0):
    #common in both finetunning and pretraining
    calc_sum_valiws(total_tokens, finetune)
    pretrained_path = root_dir+model_name+"_pretrained.pt"

    if finetune:
        path = root_dir+model_name+"_finetuned.pt"
        log = root_dir + "4_c_ft_dataloader.txt"
        checkpoint_dir = root_dir + r"ft_checkpoints"
        token_bin = root_dir + model_name + r"_Finetuning_tokens.bin"
    
    else:#pretrain
        path = pretrained_path
        log = root_dir + "3_b_dataloader.txt"
        checkpoint_dir = root_dir + r"checkpoints"
        token_bin = root_dir + model_name + r"_tokens.bin"

    if not os.path.exists(path):
        plot = create_live_loss_plot()
        dataloader = charge_dataloader(log)

        save_every = 1_000
        log_every = 10
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_chkpt, last_chkpt_index = check_last_checkpoint_file(checkpoint_dir,model_name)
        #resume_from = "Models Dev/Rosab/checkpoints/rosa_step"+str(last_chkpt)+".pt"
        if dataloader != []:
            #if input("desea cargar los datos previos al gráfico?\n*LLEVA MUCHO TIEMPO CUANDO SON DEMASIADOS DATOS.\n*NO RECOMENDADO PARA STEP 80K+\ny/n") == "y":
            for step, loss_relevante, learn_rate, elapsed in dataloader:                
                plot.update(1, step, loss_relevante, "Previous training loss", ("#FF6060"),refresh=(step % 2*log_every == 0))
                plot.update(2, step, learn_rate, "Previous lr", ("#6060FF"),refresh=(step % 2*log_every == 0))
                plot.update(3, step, elapsed, "Previous times per block", ("#60FF60"),refresh=(step % 2*log_every == 0))
                print(f"{step} = {loss_relevante} - {learn_rate} - {elapsed}")
                        
        
        multiprocessing.set_start_method("spawn", force=True)
        train(save_every, log_every, checkpoint_dir, token_bin, log, dataloader, model_name,last_chkpt,last_chkpt_index,plot, root_dir,epochs,finetune)
        plot.close()  # deja el gráfico visible al final

if __name__ == "__main__":
    root_dir = "Models Dev/RosaC/"
    model_name = "Rosa"
    train_wrapper(root_dir,model_name)#,True)




    #A REVISAR!!!
    #STEP ARRANCA SIEMPRE DESDE 0
    #EL TIEMPO SE DESMADRA AL INICIO DEL ENTRENAMIENTO (MUY ALTO) Y DESPUES SE NORMALIZA, PERO ME HACE MRD EL GRAFICO.

    1e-6