#This file reduces the english portion of the data, so that the remaining spanish portion is the biggest one.
# Also, it cleans the spanish version by length so only useful articles remain (kinda).

#Issue: spanish = 5,61 gb bruto
#       english = 22 gb bruto

#Procedimiento: filter spanish by length of articles (filter A)
#               filter english by length of articles (filter b)
#               filter english randomly until it equals 1/2 of the spanish portion size (filter c)

from datasets import load_from_disk

def filter_a_b(min_length, data = "wiki_en"):
    filtered = data.filter(
        lambda x: len(x["text"]) >= min_length,#sólo dejo las líneas con más de "min_length" caracteres.
        num_proc=4 #usa 4 procesos para mayopr velocidad
    )
    filtered = filtered.remove_columns(["id", "url", "title"])#borro los campos que no me interesan.
    return filtered


def filter_c(target_size, data):
    import random
    indices = list(range(len(data)))
    random.shuffle(indices)#mezcla los indices para que me quede una muestra un tanto homogenea de articulos.
    
    selected = []
    current_size = 0
    
    for i in indices:
        text_size = len(data[i]["text"].encode("utf-8"))#mide el articulo
        if current_size + text_size <= target_size: #si no llego al tamaño máximo pretendido
            selected.append(i)                      #agregar indice al dataset
            current_size += text_size               #conteo sencillo
        if current_size >= target_size:
            break                                   #corta si me excedí o igualé el tamaño objetivo.
    
    return data.select(selected)


def get_size(data):                                 #mide el tamaño del dataset en bytes
    total = sum(len(x["text"].encode("utf-8")) for x in data)
    return total


def save_to_file(data, path):                       #guarda el resultado en disco
    data.save_to_disk(path)     
    size = get_size(data)
    print(f"Guardado en {path} — tamaño: {size / (1024**3):.2f} GB")
    return size


def preproc_data(root = r"Models Dev/RosaB/"):
    #cargamos los datasets
    wiki_es = load_from_disk(root + r"wiki_es")
    wiki_en = load_from_disk(root + r"wiki_en")

    #chequeamos la info general de c/u
    """print(wiki_en)
    print(wiki_en.features)
    print(wiki_en[0])

    print(".\n.\n.\n.\n.\n.")

    print(wiki_es)
    print(wiki_es.features)
    print(wiki_es[0])"""

    #parte 1 (ES)
    print("Filtrando español...")
    es_filtered = filter_c(1024**3 *0.1,filter_a_b(500, wiki_es))

    es_size = save_to_file(es_filtered, root + r"wiki_es_clean")

    #parte 2 (EN)
    print("Filtrando inglés por longitud...")
    en_filtered = filter_a_b(1500, wiki_en)
    print("Reduciendo inglés a 1/2 del español...")
    en_reduced = filter_c(es_size // 2, en_filtered)
    save_to_file(en_reduced, root + r"wiki_en_clean")

    print("Listo.")


if __name__ == "__main__":
    preproc_data()