from datasets import load_dataset
from huggingface_hub import login

def get_datasets(root_dir):
    lgin = input("Pegue su token de login de HuggingFace\n")

    login(lgin)#CREDENCIALES DE HUGGINGFACE PARA PODER BAJAR LOS DATASETS


    # Wikipedia español
    ds_es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
    ds_es.save_to_disk(root_dir + r"/wiki_es")

    # Wikipedia inglés
    ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    ds_en.save_to_disk(root_dir + r"/wiki_en")


if __name__=="__main__":
    root_dir = r"Models Dev/Rosa"
    get_datasets(root_dir)