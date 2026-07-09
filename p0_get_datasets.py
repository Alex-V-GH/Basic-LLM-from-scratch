from datasets import load_dataset
from huggingface_hub import login
import os

def get_datasets(root_dir):

    if not os.path.exists(root_dir + r"/wiki_es") or not os.path.exists(root_dir + r"/wiki_en"):
        login(input("Pegue su token de login de HuggingFace\n"))

    # Wikipedia español
    
    if not os.path.exists(root_dir + r"/wiki_es"):
        ds_es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
        ds_es.save_to_disk(root_dir + r"/wiki_es")

    # Wikipedia inglés
    if not os.path.exists(root_dir + r"/wiki_en"):
        ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        ds_en.save_to_disk(root_dir + r"/wiki_en")


if __name__=="__main__":
    root_dir = r"Models Dev/Rosa"
    get_datasets(root_dir)