from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import os

def procesar_oasst2(ds, traducir, ejemplos):
    print("\n=== Procesando OASST2 ES + EN→ES ===")

    # construir índice message_id → mensaje
    index = {row["message_id"]: row for row in ds}

    contador = 0
    for msg in ds:
        # solo mensajes de asistente, no borrados, con review positivo
        if msg["role"] != "assistant":
            continue
        if msg["deleted"]:
            continue
        if not msg["review_result"]:
            continue
        if msg["lang"] not in ("es", "en"):
            continue

        # buscar el padre (prompter)
        parent = index.get(msg["parent_id"])
        if parent is None:
            continue
        if parent["role"] != "prompter":
            continue
        if parent["deleted"]:
            continue

        pregunta  = parent["text"].strip()
        respuesta = msg["text"].strip()

        if not pregunta or not respuesta:
            continue

        # traducir si es EN
        if msg["lang"] == "en":
            pregunta  = traducir([pregunta])[0]
            respuesta = traducir([respuesta])[0]

        ejemplos.append(f"<bos><user>: {pregunta}<mask><pregunta><rosa>: {respuesta}<eos>")
        contador += 1
        if contador % 500 == 0:
            print(f"OASST2: {contador} pares procesados")

    print(f"OASST2 total: {contador} pares")


def procesar(root_dir):
    output_path = root_dir + r"finetuning_conversations.txt"
    if not os.path.exists(output_path):

        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer  = MarianTokenizer.from_pretrained(model_name)
        model      = MarianMTModel.from_pretrained(model_name)

        def traducir(textos: list[str]) -> list[str]:
            batch = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**batch)
            return tokenizer.batch_decode(translated, skip_special_tokens=True)

        ejemplos = []
        contador_a = 0
        contador_b = 0
        contador_c = 0

        # ── Dataset 1: everyday-conversations (~8625 pares) ──────────────
        print("=== Procesando everyday-conversations ===")
        ds_everyday = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")

        for conv in ds_everyday:
            mensajes = conv["messages"]
            pares = []
            for i in range(0, len(mensajes) - 1, 2):
                if mensajes[i]["role"] == "user" and mensajes[i+1]["role"] == "assistant":
                    contador_a += 1
                    pares.append((mensajes[i]["content"], mensajes[i+1]["content"]))

            if not pares:
                continue

            usuarios_es   = traducir([p[0] for p in pares])
            asistentes_es = traducir([p[1] for p in pares])

            for u, a in zip(usuarios_es, asistentes_es):
                ejemplos.append(f"<bos><user>: {u}<mask><libre><rosa>: {a}<eos>")
                contador_b += 1
            contador_c += 1
            print(f"Pares Identificados={contador_a} | Procesados={contador_b} | Conversaciones={contador_c}")

        # ── Dataset 2: dolly ES (~15015 pares) ───────────────────────────
        print("\n=== Procesando dolly ES ===")
        ds_dolly_es = load_dataset("argilla/databricks-dolly-15k-curated-multilingual", split="es")

        for i, row in enumerate(ds_dolly_es):
            instruccion = row["instruction"].strip()
            respuesta   = row["response"].strip()
            if instruccion and respuesta:
                ejemplos.append(f"<bos><user>: {instruccion}<mask><pregunta><rosa>: {respuesta}<eos>")
                contador_b += 1
            if i % 1000 == 0:
                print(f"Dolly ES: {i}/{len(ds_dolly_es)}")

        # ── Dataset 3: dolly EN traducido (~15015 pares) ─────────────────
        print("\n=== Procesando dolly EN → ES ===")
        ds_dolly_en = load_dataset("argilla/databricks-dolly-15k-curated-multilingual", split="en")

        for i, row in enumerate(ds_dolly_en):
            instruccion = row["instruction"].strip()
            respuesta   = row["response"].strip()
            if not instruccion or not respuesta:
                continue

            instruccion_es = traducir([instruccion])[0]
            respuesta_es   = traducir([respuesta])[0]
            ejemplos.append(f"<bos><user>: {instruccion_es}<mask><pregunta><rosa>: {respuesta_es}<eos>")
            contador_b += 1
            if i % 100 == 0:
                print(f"Dolly EN→ES: {i}/{len(ds_dolly_en)} | Total pares: {contador_b}")

        # ── Dataset 4: OASST2 ES + EN traducido ──────────────────────────
        ds_oasst2 = load_dataset("OpenAssistant/oasst2", split="train")
        procesar_oasst2(ds_oasst2, traducir, ejemplos)

        # ── Guardar ───────────────────────────────────────────────────────
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ejemplos))

        print(f"\nTotal ejemplos: {len(ejemplos)}")
        print(f"Guardado en {output_path}")

if __name__ == "__main__":
    root_dir = "Models Dev/Rosab/"
    procesar(root_dir)