from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
tokenizer  = MarianTokenizer.from_pretrained(MODEL_NAME)
model      = MarianMTModel.from_pretrained(MODEL_NAME)

def traducir(textos: list[str]) -> list[str]:
    batch = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**batch)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def procesar(output_path=r"Models Dev/Rosab/finetuning_conversations.txt"):
    ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
    contador_a= 0
    contador_b= 0
    contador_c= 0
    ejemplos = []
    for conv in ds:
        mensajes = conv["messages"]
        pares = []
        for i in range(0, len(mensajes) - 1, 2):
            if mensajes[i]["role"] == "user" and mensajes[i+1]["role"] == "assistant":
                contador_a +=1
                print(f"Pares Identificados={contador_a} | Pares Procesados={contador_b} | Conversacion={contador_c}")
                pares.append((mensajes[i]["content"], mensajes[i+1]["content"]))

        if not pares:
            continue

        usuarios   = [p[0] for p in pares]
        asistentes = [p[1] for p in pares]

        usuarios_es   = traducir(usuarios)
        asistentes_es = traducir(asistentes)

        for u, a in zip(usuarios_es, asistentes_es):
            ejemplos.append(f"<bos>[usuario]: {u}<mask>[libre][rosa]: {a}<eos>")
            contador_b +=1
            print(f"Pares Identificados={contador_a} | Pares Procesados={contador_b} | Conversacion={contador_c}")
        contador_c +=1
        print(f"Pares Identificados={contador_a} | Pares Procesados={contador_b} | Conversacion={contador_c}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ejemplos))

    print(f"Total ejemplos: {len(ejemplos)}")
    print(f"Guardado en {output_path}")

if __name__ == "__main__":
    procesar()