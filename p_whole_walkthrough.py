from live_loss_plot import create_live_loss_plot

from p0_get_datasets import get_datasets
from p1_preprocess_data import preproc_data
from p2_train_tokenizer import train_tokenizer_root
from p3_a_tokenize_datasets import build_token_bin
from p3_b_train_weights import train_wrapper
from p3_c_model_test import test_model
from p4_a_finetuning_dataset_gather import procesar
from p4_b_finetuning_dataset_additions_and_tokenization import crear_dataset,tokenizar_dataset

def titulo(titul):
    print("===========================================================\n===========================================================\n",
    f"                    {titul}\n",
    "===========================================================\n===========================================================\n",)
if __name__ == "__main__":
    root_dir = r"C:\Users\Alex\Desktop\Violetta AI\Models Dev\RosaC/" #input("Pegue el directorio raiz de su proyecto de modelo:\n") + "/"
    model_name = "Rosa"#input("Cómo va a nombrar a su modelo?\n")
    titulo("Fase 0: Obteniendo los datasets principales base.")
    get_datasets(root_dir)
    #Result:    root_dir + r"/wiki_en" // root_dir + r"/wiki_es"

    titulo("Fase 1: Preprocesando los datos del dataset principal")
    preproc_data(1024**3 *0.1, root_dir)
    #Result:    root_dir + r"wiki_es_clean" // root_dir + r"wiki_en_clean"
    #  
    titulo("Fase 2: Entrenando el tokenizador.")
    train_tokenizer_root(root_dir,model_name)
    #Result:    root_dir + model_name + r"_tokenizer" + r".json"

    titulo("Fase 3 A: Tokenizando el dataset principal.")
    total_tokens = build_token_bin(root_dir, model_name,5_000_000,1000,2)
    #Result:    root_dir + model_name + r"_tokens.bin" // Return= Total de tokens del dataset
    titulo("Fase 3 B: Preentrenamiento del modelo.")
    train_wrapper(root_dir, model_name, False, total_tokens)
    #Result:    root_dir + model_name + "_pretrained.pt"
    titulo("Fase 3 C: Un pequeño test.")
    #test_model(root_dir, model_name, pretrained = 1, finetuned = None)
    #Result: None

    titulo("Fase 4 A: Obtencion de datasets genéricos de conversacion.")
    procesar(root_dir)
    #result: root_dir + r"ft datas/"
    titulo("Fase 4 B1: Creación de pares personalizados para el dataset de finetuning.")
    crear_dataset(root_dir)
    #result: root_dir + r"ft datas/finetuning_conversations.txt" (RE)
    titulo("Fase 4 B2: Tokenización del dataset para finetunning.")
    ft_tokens = tokenizar_dataset(root_dir, model_name) #25,215,157 tokens
    #result: root_dir + r"Finetuning_tokens.bin"
    titulo("Fase 4 C: Finetunning.")
    train_wrapper(root_dir,model_name,finetune = True,total_tokens = ft_tokens, epochs = 15)
    #result: root_dir + model_name + "_finetuned.pt"
    titulo("Fase 4 EXTRA: Otro pequeño test.")
    test_model(root_dir, model_name, pretrained = None, finetuned = 1)
    #Result: None