import multiprocessing

from p0_get_datasets import get_datasets
from p1_preprocess_data import preproc_data
from p2_train_tokenizer import train_tokenizer_root
from p3_a_tokenize_datasets import build_token_bin
from p3_b_train_weights import train, charge_dataloader
from live_loss_plot import create_live_loss_plot
from p3_c_model_test import test_model

root_dir = input("Pegue el directorio raiz de su proyecto de modelo:\n")
model_name = input("Cómo va a nombrar a su modelo?")


"""Fase 0 — obtener datasets"""
get_datasets(root_dir)
#Result:    root_dir + r"/wiki_en" // root_dir + r"/wiki_es"


"""Fase 1 — Dataset :Limpiar y filtrar Wikipedia ES y EN. ES completo, EN reducido a la mitad del tamaño de ES. Borrar columnas id, urly title. conservar solo text."""
preproc_data(root_dir)
#Result:    root_dir + r"wiki_es_clean" // root_dir + r"wiki_en_clean" 


"""Fase 2 — Tokenizador :Entrenar un tokenizador propio sobre tu corpus. Vocabulario en español primero, inglés secundario. Incluye caracteres en chino, japonés, letras raras, emojis, etc. Debe ser capaz de leer todo. Debe ser capaz de contestar sólo en castellano o romanji."""
train_tokenizer_root(root_dir,model_name)
#Result:    root_dir + model_name + r"_tokenizer" + r".json"

build_token_bin(root_dir, model_name)
#Result:    root_dir + model_name + r"_tokens.bin"


"""Fase 3 — Preentrenamiento: Arquitectura transformer 50M desde cero. Entrenamiento sobre el corpus limpio para que aprenda lenguaje."""
plot = create_live_loss_plot()
dataloader = charge_dataloader(root_dir+"Models Dev/Rosa/3_b_dataloader.txt")
for step, loss in dataloader:
    plot.update(step, loss)
    #este contador es para acelerar en numeros altos de steps tomados.
    contador+=1
multiprocessing.set_start_method("spawn", force=True)
train()#Poner los argumentos correctos al terminar de entrenar
plot.close()
#Result:

test_model() #maybe integrar a p3_b(?
#Result: None


"""Fase 4 — Fine-tuning conversacional: Dataset conversacional que vamos a construir o conseguir. Acá es donde el modelo aprende el flujo que describiste (clasificar input → rama → responder).
En esta fase se agrega reconocimiento de los caracteres y palabras básicas de idiomas no entrenados. No necesita saber traducir idiomas que no sean inglés. Sólo poder reconocerlos, como máximo.
 Y sus equivalencias fonéticas aproximadas al español (opcional)"""
"""Fase 5 — Memoria externa: Integración con ChromaDB para la memoria de usuario extensa. (en este punto, plantear las otras posibles db. NO AHORA, sólo al llegar a esta fase."""
"""Fase 6 — Flujo de decisión: El código Python que orquesta todo."""
