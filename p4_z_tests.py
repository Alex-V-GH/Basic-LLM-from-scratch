from datasets import load_dataset

def try_somoschat():
    ds = load_dataset("somos-nlp/somos-chat", split="train")
    print(ds)
    print(ds.features)
    print(ds[0])

if "__name__" == "__main__":
    try_somoschat()