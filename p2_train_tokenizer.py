from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_from_disk
import os

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "<sep>", "<user>", "<rosa>", 
#   |                           BASICOS NECESARIOS        |     Entidades     | 
    "<orden>","<pregunta>", "<pregunta imposible>", "<libre>"
#                    Clasificacion                          |
]

#Agregar: Emociones permanentes, emociones de corta duracion, acciones del vroid (aparece, escondete, sorda, escucha, etc), acciones del sistema (apagate, esconde la ventana, etc)
GUARANTEED_TOKENS = [
    # especiales del proyecto
    "💜", "🌹", "💎",
    # caritas
    "😀","😁","😂","🤣","😃","😄","😅","😆","😇","😉","😊","🙂","🙃","😋","😌","😍","🥰","😘","😗","😙","😚","🤩","😏","😒","😞","😔","😟","😕","🙁","☹️","😣","😖","😫","😩","🥺","😢","😭","😤","😠","😡","🤬","🤯","😳","🥵","🥶","😱","😨","😰","😥","😓","🤗","🤔","🤭","🤫","🤥","😶","😐","😑","😬","🙄","😯","😦","😧","😮","😲","🥱","😴","🤤","😪","😵","🤐","🥴","🤢","🤮","🤧","😷","🤒","🤕","🤑","😈","👿","💀","☠️","🤡","👻","👽","🤖","💩",
    # manitas
    "👋","🤚","🖐️","✋","🖖","👌","🤌","🤏","✌️","🤞","🤟","🤘","🤙","👈","👉","👆","🖕","👇","☝️","👍","👎","✊","👊","🤛","🤜","👏","🙌","👐","🤲","🤝","🙏","✍️","💅",
    # corazones
    "❤️","🧡","💛","💚","💙","💜","🖤","🤍","🤎","💔","❣️","💕","💞","💓","💗","💖","💘","💝","💟","❤️‍🔥","❤️‍🩹",
    # corona
    "👑",
    # luna y sol
    "🌑","🌒","🌓","🌔","🌕","🌖","🌗","🌘","🌙","🌚","🌛","🌜","🌝","🌞","☀️","🌤️","⛅","🌥️",
    # tiempo y clima
    "🌈","🌂","☂️","🌧️","⛈️","🌩️","🌨️","❄️","☃️","⛄","🌬️","💨","🌪️","🌫️","🌦️","🌟","⭐","🌠","⚡","🌡️","☁️",
    # elementos
    "🔥","💧","💦","🌊","🧊","💨","🌬️","⚗️","🌋","🌿","🍃",
    # señales y símbolos
    "☢️","☣️","🔴","🟠","🟡","🟢","🔵","🟣","🟤","⚫","⚪","✅","❎","✔️","❌","⭕","🚫","⚠️","🩺","💊","🧬","⚙️","🔒","🔓","🗡️","🛡️","⚔️",
    # cirílico ruso + ucraniano (minúsculas)
    "а","б","в","г","ґ","д","е","ё","є","ж","з","и","і","ї","й","к","л","м","н","о","п","р","с","т","у","ф","х","ц","ч","ш","щ","ъ","ы","ь","э","ю","я",
    # hiragana
    "あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん",
    # katakana
    "ア","イ","ウ","エ","オ","カ","キ","ク","ケ","コ","サ","シ","ス","セ","ソ","タ","チ","ツ","テ","ト","ナ","ニ","ヌ","ネ","ノ","ハ","ヒ","フ","ヘ","ホ","マ","ミ","ム","メ","モ","ヤ","ユ","ヨ","ラ","リ","ル","レ","ロ","ワ","ヲ","ン",
    # árabe
    "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي",
    # hangul (30 jamos)
    "ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ","ㅏ","ㅑ","ㅓ","ㅕ","ㅗ","ㅛ","ㅜ","ㅠ","ㅡ","ㅣ","ㄲ","ㄸ","ㅃ","ㅆ","ㅉ",
    # chino (50 más frecuentes)
    "的","一","是","在","不","了","有","和","人","这","中","大","为","上","个","国","我","以","要","他","时","来","用","们","生","到","作","地","于","出","就","分","对","成","会","可","主","发","年","动","同","工","也","能","下","过","子","说","产","种",
]

def batch_iterator(dataset_es, dataset_en, batch_size=1000):
    for i in range(0, len(dataset_es), batch_size):
        yield dataset_es[i:i+batch_size]["text"]
    for i in range(0, len(dataset_en), batch_size):
        yield dataset_en[i:i+batch_size]["text"]

def train_tokenizer(dataset_es, dataset_en, output_path=r"Models Dev/RosaB/rosa_tokenizer", vocab_size=32000,):

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=GUARANTEED_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        batch_iterator(dataset_es, dataset_en),
        trainer=trainer,
    )

    tokenizer.save(f"{output_path}.json")
    print(f"Tokenizador guardado en {output_path}.json")
    print(f"Vocabulario final: {tokenizer.get_vocab_size()} tokens")
    return tokenizer

def train_tokenizer_root(root = r"Models Dev/RosaB/", model_name = r"Rosa"):
    if not os.path.exists(root+model_name+"_tokenizer"):

        print("Cargando datasets...")
        wiki_es = load_from_disk(root + r"wiki_es_clean")
        wiki_en = load_from_disk(root + r"wiki_en_clean") 

        print("Entrenando tokenizador...")
        tokenizer = train_tokenizer(wiki_es, wiki_en,root+model_name+"_tokenizer")

        test = tokenizer.encode(input("Con qué querés probar el tokenizador? Dame un texto de ejemplo."))
        print("Test encode:", test.tokens)
        print("Test decode:", tokenizer.decode(test.ids))


if __name__ == "__main__":
    train_tokenizer_root()