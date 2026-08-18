from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import os


def guardar (path,pares):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(pares))
    print(f"\nTotal ejemplos: {len(pares)}")
    print(f"Guardado en {path}")


def procesar(root_dir):
    output_folder = root_dir + r"ft datas/"
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder + r"everyday_conversations_llama.txt"
    print(output_path)
    ejemplos = []
    contador_a = 0
    contador_b = 0
    contador_c = 0


    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer  = MarianTokenizer.from_pretrained(model_name)
    model      = MarianMTModel.from_pretrained(model_name)

    def traducir(textos: list[str]) -> list[str]:
        batch = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**batch)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)

    # ── Dataset 1: everyday-conversations (~8625 pares) ────────────── (unos minutitos)
    if not os.path.exists(output_path):
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

        guardar(output_path, ejemplos)


    # ── Dataset 2: dolly ES (~15015 pares) ─────────────────────────── (unos segundos)
    output_path = output_folder + r"databricks_dolly_es.txt"
    ejemplos = []#wipe var
    if not os.path.exists(output_path):
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
        guardar(output_path, ejemplos)


        # ── Dataset 3: dolly EN traducido (~15015 pares) ─────────────────(20 hs aprox.)
    output_path = output_folder + r"databricks_dolly_en.txt"
    ejemplos = []#wipe var
    if not os.path.exists(output_path):
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
        guardar(output_path, ejemplos)

    # ── Dataset 4: OASST2 ES + EN traducido (~??? pares. aprox. 1h20' per block (769 pares) aprox 7 segs per par)──────────────────────────
    #14 k ejemplos??? CORROBORAR!
    output_folder_oasst = output_folder + r"oasst2/" #la uso como base para este caso
    os.makedirs(output_folder_oasst, exist_ok=True)# me aseguro de que exista
    try:
        parsed = max(
            int(os.path.splitext(f)[0])
            for f in os.listdir(output_folder_oasst)
            if os.path.isfile(os.path.join(output_folder_oasst, f))
        )
        print(f"se encontro un total de {parsed} pares parseados")
        parsed += 1
    except:
        parsed = 0
        print("parsed es 0")
    output_path = output_folder_oasst + str(parsed) + ".txt"
    ds_oasst2 = load_dataset("OpenAssistant/oasst2", split="train")
    oasst_length=len(ds_oasst2)
    #loop de contar, parsear, agregar, guardar:
    if not parsed >= oasst_length:
        buffer_len = 2_000# (4k pares por vez me parece razonable en tiempo. En todo caso, MEDIR)
        ejemplos = []#wipe var
        print(f"\n=== Procesando OASST2 ES + EN→ES desde posición {parsed}===")
        # construir índice message_id → mensaje

        index = {row["message_id"]: row for row in ds_oasst2}
        contador = 0
        contador2 = 0
        for msg in ds_oasst2:
            if contador + contador2 >= parsed-1:
            # solo mensajes de asistente, no borrados, con review positivo
                if msg["role"] != "assistant":
                    contador2 += 1
                    continue
                if msg["deleted"]:
                    contador2 += 1
                    continue
                if not msg["review_result"]:
                    contador2 += 1
                    continue
                if msg["lang"] not in ("es", "en"):
                    contador2 += 1
                    continue

                # buscar el padre (prompter)
                parent = index.get(msg["parent_id"])
                if parent is None:
                    contador2 += 1
                    continue
                if parent["role"] != "prompter":
                    contador2 += 1
                    continue
                if parent["deleted"]:
                    contador2 += 1
                    continue

                pregunta  = parent["text"].strip()
                respuesta = msg["text"].strip()

                if not pregunta or not respuesta:
                    contador2 += 1
                    continue

                # traducir si es EN
                if msg["lang"] == "en":
                    pregunta  = traducir([pregunta])[0]
                    respuesta = traducir([respuesta])[0]

                ejemplos.append(f"<bos><user>: {pregunta}<mask><pregunta><rosa>: {respuesta}<eos>")
                if (contador + 1) % 50 == 0:
                    print(f"OASST2: {contador2} pares procesados, {contador} pares parseados. Total de {contador + contador2}")
                if (contador + contador2 + 1) % buffer_len == 0:
                    output_path = output_folder_oasst + str(contador + contador2 + 1) + ".txt"#actualiz. path
                    guardar(output_path,ejemplos)
                    ejemplos = []#reset buffer
                contador += 1
            else:
                contador2 +=1
    try:
        guardar(output_folder_oasst + str(contador + contador2), ejemplos)
    except Exception as e:
        print(e)
        pass
    try:
        print(f"OASST2: {contador2} pares procesados, {contador} pares parseados. Total de {contador + contador2}\nFIN.")
    except Exception as e:
        print (e)





if __name__ == "__main__":
    root_dir = "Models Dev/Rosac/"
    procesar(root_dir)



    #IMPORTANTE"""!!!!!!!!!!!!!!!!!!!!!!!!!
    #SUMAR AL CONTADOR <<<<ANTES>>>>