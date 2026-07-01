import torch
from tokenizers import Tokenizer

import p3_b_train_weights
# ── config ────────────────────────────────────────────────────────
MODEL_PATH     = r"Models Dev/Rosa/checkpoints/rosa_step215000.pt"
TOKENIZER_PATH = r"Models Dev/Rosa/rosa_tokenizer.json"
CONTEXT_LEN    = 1024
VOCAB_SIZE     = 32000
MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.8
TOP_K          = 50
# ─────────────────────────────────────────────────────────────────


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x   = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -CONTEXT_LEN:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature

            # top-k
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float("-inf")

            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x       = torch.cat([x, next_id], dim=1)

    generated = x[0, len(ids):].tolist()
    return tokenizer.decode(generated)

def test_model():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    model = p3_b_train_weights.Rosa().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == "exit":
            break
        output = generate(model, tokenizer, prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, device)
        print(f"\nRosa: {output}")


if __name__ == "__main__":
    test_model()