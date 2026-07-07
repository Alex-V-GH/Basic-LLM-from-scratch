import numpy as np
from tokenizers import Tokenizer
from datasets import load_from_disk
import random

# ── config ────────────────────────────────────────────────────────
DTYPE          = np.uint16   # alcanza para vocab ≤ 65535 (tu vocab=32000 ✓)
FLUSH_EVERY    = 5_000_000   # tokens por flush a disco
LOG_EVERY      = 1000
DATASETS_VERSIONES = 2  #cantidad de "shuffles" del dataset completo. Esto se hace para que el modelo aprenda
                        #inglés y castellano en simultáneo, y que las distintas pasadas no sean sobre el mismo
                        #orden (evita overfitting). Como regla general, DATASETS_VERSIONES = training epochs
# ─────────────────────────────────────────────────────────────────
#MODIFICAR PARA PODER LLAMAR DESDE 4A
def build_token_bin(root_dir,model_name):
    TOKENIZER_PATH = root_dir + model_name + r"_tokenizer.json"
    WIKI_ES_PATH   = root_dir + r"wiki_es_clean"
    WIKI_EN_PATH   = root_dir + r"wiki_en_clean"
    OUT_BIN        = root_dir + model_name + r"_tokens.bin"

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id    = tokenizer.token_to_id("<eos>")

    print("Cargando datasets...")
    wiki_es = load_from_disk(WIKI_ES_PATH)
    wiki_en = load_from_disk(WIKI_EN_PATH)

    n_es = len(wiki_es)
    n_en = len(wiki_en)

    # índices: 0..n_es-1 son ES, n_es..n_es+n_en-1 son EN
    all_indices = list(range(n_es + n_en))

    total_tokens = 0
    buf = []

    with open(OUT_BIN, "wb") as f:
        for version in range(DATASETS_VERSIONES):
            print(f"\n── Versión {version+1}/{DATASETS_VERSIONES} ──")
            random.seed(version)
            random.shuffle(all_indices)

            for i, idx in enumerate(all_indices):
                if idx < n_es:
                    text = wiki_es[idx]["text"]
                else:
                    text = wiki_en[idx - n_es]["text"]

                ids = tokenizer.encode(text).ids
                buf.extend(ids)
                buf.append(eos_id)

                if len(buf) >= FLUSH_EVERY:
                    arr = np.array(buf, dtype=DTYPE)
                    f.write(arr.tobytes())
                    total_tokens += len(buf)
                    buf = []

                if i % LOG_EVERY == 0:
                    print(f"  art {i:,} | tokens escritos: {total_tokens:,}")

        if buf:
            arr = np.array(buf, dtype=DTYPE)
            f.write(arr.tobytes())
            total_tokens += len(buf)

    size_gb = total_tokens * 2 / 1e9
    print(f"\nListo: {OUT_BIN}")
    print(f"Total tokens : {total_tokens:,} \n    ({total_tokens/DATASETS_VERSIONES} tokens por pasada)")
    print(f"Tamaño disco : {size_gb:.2f} GB")

if __name__ == "__main__":
    root_dir = r"Models Dev/Rosab/"
    model_name = r"Rosa"
    build_token_bin(root_dir,model_name)