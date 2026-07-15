from live_loss_plot import create_live_loss_plot

from p0_get_datasets import get_datasets
from p1_preprocess_data import preproc_data
from p2_train_tokenizer import train_tokenizer_root
from p3_a_tokenize_datasets import build_token_bin
from p3_b_train_weights import train_wrapper
from p3_c_model_test import test_model
from p4_a_finetuning_dataset_gather import procesar
from p4_b_finetuning_dataset_additions_and_tokenization import crear_dataset,tokenizar_dataset
from p4_c_finetune import finetune

def titulo(titul):
    print("===========================================================\n===========================================================\n",
    f"                    {titul}\n",
    "===========================================================\n===========================================================\n",)
if __name__ == "__main__":
    root_dir = input("Pegue el directorio raiz de su proyecto de modelo:\n") + "/"
    model_name = input("Cómo va a nombrar a su modelo?\n")

    #if os.path.exists(token_bin)

    """Fase 0 — obtener datasets"""
    titulo("Fase 0: Obteniendo los datasets principales base.")
    get_datasets(root_dir)
    #Result:    root_dir + r"/wiki_en" // root_dir + r"/wiki_es"


    """Fase 1 — Dataset :Limpiar y filtrar Wikipedia ES y EN. ES completo, EN reducido a la mitad del tamaño de ES. Borrar columnas id, urly title. conservar solo text."""
    titulo("Fase 1: Preprocesando los datos del dataset principal")
    preproc_data(1024**3 *0.1, root_dir)
    #Result:    root_dir + r"wiki_es_clean" // root_dir + r"wiki_en_clean" 


    """Fase 2 — Tokenizador :Entrenar un tokenizador propio sobre tu corpus. Vocabulario en español primero, inglés secundario. Incluye caracteres en chino, japonés, letras raras, emojis, etc. Debe ser capaz de leer todo. Debe ser capaz de contestar sólo en castellano o romanji."""
    titulo("Fase 2: Entrenando el tokenizador.")
    train_tokenizer_root(root_dir,model_name)
    #Result:    root_dir + model_name + r"_tokenizer" + r".json"


    """Fase 3 — Preentrenamiento: Arquitectura transformer 50M desde cero. Entrenamiento sobre el corpus limpio para que aprenda lenguaje."""
    titulo("Fase 3 A: Tokenizando el dataset principal.")
    total_tokens = build_token_bin(root_dir, model_name,5_000_000,1000,2)
    #Result:    root_dir + model_name + r"_tokens.bin" // Return= Total de tokens del dataset

    titulo("Fase 3 B: Preentrenamiento del modelo.")
    train_wrapper(root_dir, model_name, total_tokens)
    #Result:    root_dir + model_name + "_pretrained.pt"

    titulo("Fase 3 C: Un pequeño test.")
    test_model(root_dir, model_name, pretrained = 1, finetuned = None)
    #Result: None


    """Fase 4 — Fine-tuning conversacional: Dataset conversacional que vamos a construir o conseguir. Acá es donde el modelo aprende el flujo que describiste (clasificar input → rama → responder).
    En esta fase se agrega reconocimiento de los caracteres y palabras básicas de idiomas no entrenados. No necesita saber traducir idiomas que no sean inglés. Sólo poder reconocerlos, como máximo.
    Y sus equivalencias fonéticas aproximadas al español (opcional)"""
    """input -->decide qué tipo de input es --> si es pregunta --> si la puede manejar --> generación de respuesta
    si no puede manejar la pregunta --> generación de respuesta fragmentada en formato específico para post-procesamiento
    si no es pregunta, pero es orden según directivas específicas -->  generación de respuesta (cumplir la orden, normalmente comunicar algo)
    si no es pregunta ni orden --> generación de respuesta (manejar la situación según parámetros pasados como parte del prompt)
    no quiero que SEPA cosas grandes, sino que tenga una cultura general mínima y razonable, cosas 
    básicas como poder decir "sí, 1 es un número." o "what did you say? está en inglés y significa qué dijiste?",
    sumado a que pueda adaptarse al usuario con una memoria de contexto, esta sí, quizás extensa. (qué le 
    gusta al usuario, cómo este trata al agente de IA, qué cosas NO le gustan al usuario, cierto sentido 
    básico de personalidad/rol, etc)"""
    titulo("Fase 4 A: Obtencion de datasets genéricos de conversacion.")
    procesar(root_dir)
    #result: root_dir + r"finetuning_conversations.txt"

    titulo("Fase 4 B1: Creación de pares personalizados para el dataset de finetuning.")
    crear_dataset(root_dir)
    #result: root_dir + r"finetuning_conversations.txt" (RE)
    titulo("Fase 4 B2: Tokenización del dataset para finetunning.")
    tokenizar_dataset(root_dir, model_name)
    #result: root_dir + r"Finetuning_tokens.bin"

    titulo("Fase 4 C: Finetunning.")
    finetune(root_dir, model_name, 25, 10)#AGREGAR GRAFICO DESPUES!!! // Reemplazar por el metodo p3_b_train_weights.
    #result: root_dir + model_name + "_finetuned.pt"

    titulo("Fase 4 EXTRA: Otro pequeño test.")
    test_model(root_dir, model_name, pretrained = None, finetuned = 1)
    #Result: None



    """Fase 5 — Memoria externa: Integración con ChromaDB para la memoria de usuario extensa. (en este punto, plantear las otras posibles db. NO AHORA, sólo al llegar a esta fase."""
    """Fase 6 — Flujo de decisión: El código Python que orquesta todo."""
