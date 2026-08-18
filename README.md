# **Basic LLM From Scratch**   or   ***Road to Rosa🌹***

## Introduction:
This project comes from needing an llm agent to integrate in Violetta (check my other repos). I tried to use some of the most common, but none of them was able to cover the need.
So it all began: I tried to finetune some specific model, but failed due to not having too much information about it´s architecture. 
Then i thought of making this little piece which i would have absolute (or at least much more) control over it.
I defined the **goals** for this project, which are: 
- It HAS to be a language model. Moreso, a LARGE LANGUAGE MODEL.
- It HAS to understand some specific orders given by the Violetta pipeline, and act according to them.
- It HAS to be relatively light, so it runs in most computers while the users can still use them.
- I HAVE to know EXACTLY the architecture of it, and change it as Violetta requires so.
- It HAS to "know" spanish as it´s main language, and english as secondary.
- It HAS to have conversational capability
- It HAS to manage context dynamically
- It HAS to learn some things from the user, and sometimes finetune itself to integrate that
- It HAS to know wether is the prompt an order, a conversation, a question, etc, as Violetta requires so.

## Current Status
Currently trying to finish the main pipeline with flawed results, but pointing to the goals

## Architecture
*To be filled*

## Usage
Step by step:
1. Download python 3.11
2. Download this repository
3. (**optional**) on a terminal, create a virtual environment with `python -m venv venv`
4. On a terminal use the following command to setup your environment: `pip install requirements.txt`
5. Please check:
    - You need a HuggingFace token to download the first databases
    - Inside **p_whole_walkthrough.py** you will find lines **17** and **18**, which i recommend to change.
        - Line 17 has the path which will be used to put all the necesary files. Please make sure to set it correctly!
        - Line 18 is the model name. You can change it to whatever you like, *but i would not use other chars than letters*
    - You probably should also check the finetuning dataset creator on **p4_b_finetuning_dataset_additions_and_tokenization.py**
6. Run **p_whole_walkthrough** and enjoy the proccess.

**I STRONGLY RECOMMEND TO CHECK THE FULL CODE TO LEARN HOW TO SET AN LLM FROM ZERO**, but do as you wish.

## Results
- *root_dir + r"/wiki_en" // root_dir + r"/wiki_es"*            Raw pretraining datasets
- *root_dir + r"wiki_es_clean" // root_dir + r"wiki_en_clean"*  Processed pretraining datasets (cut them to a specific size, basically)
- *root_dir + model_name + r"_tokenizer.json"*                  Tokenizer
- *root_dir + model_name + r"_tokens.bin"*                      Processed pretraining datasets, now tokenized
- *root_dir + model_name + "checkpoints"*                       Checkpoints of pretraining, so you can close it an resume later
- *root_dir + model_name + "_pretrained.pt"*                    First model output. Useless, but it now has the languages to some extent.
- *root_dir + r"ft datas/"*                                     Finetunning datasets from internet, most of them translated to spanish.
- *root_dir + r"ft datas/finetuning_conversations.txt"*         Custom finetunning dataset
- *root_dir + r"Finetuning_tokens.bin"*                         All finetuning datasets, now tokenized
- *root_dir + model_name + "ft_checkpoints"*                    Checkpoints of finetuning, so you can close it an resume later
- *root_dir + model_name + "_finetuned.pt"*                     Finetuned model
