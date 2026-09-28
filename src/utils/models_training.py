import datetime
import json
import os
from tqdm import tqdm

import sys

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()  

from gpt2 import gpt2_faster_train
from llama import llama_lora_faster_train_j


TRAINING_RUN_ID = f"{datetime.datetime.now():%Y%m%d_%H%M%S_%f}_pid{os.getpid()}"


def temporary_training_paths(base_output_dir):
    parent_dir = os.path.dirname(base_output_dir)
    output_name = os.path.basename(os.path.normpath(base_output_dir))
    temp_root = os.path.join(parent_dir, ".training_tmp", TRAINING_RUN_ID)
    return temp_root, os.path.join(temp_root, output_name), os.path.join(temp_root, "args.json")


def train_model(model_type, model_name, max_epochs, batch_size, train_set_file_path, base_output_dir, loo_folder=None, save_every=5, gradient_checkpointing=True):
    """
    Addestra il modello salvando checkpoint incrementali
    
    Args:
        max_epochs: Numero totale di epoche
        train_set_file_path: Percorso al file di addestramento
        base_output_dir: Cartella base di output (senza suffisso _Xep)
        save_every: Salva il modello ogni queste epoche
        gradient_checkpointing: Abilita gradient checkpointing per ridurre memoria
    """
    
    temp_root, temporary_output_dir, arguments_file = temporary_training_paths(base_output_dir)

    args = {
        "output_dir": temporary_output_dir,
        "model_name_or_path": model_name,
        "do_train": "y",
        "train_file": train_set_file_path,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": max_epochs,
        "remove_unused_columns": False,
        "seed": 42,
        "gradient_checkpointing": gradient_checkpointing,
    }

    # Crea la directory se non esiste
    os.makedirs(temp_root, exist_ok=True)
    
    # Scrivi file argomenti
    with open(arguments_file, 'w') as f:
        json.dump(args, f)

    try:
        if model_type == 'gpt2':
            # Avvia addestramento con il file di parametri
            gpt2_faster_train.train_model(
                ["", arguments_file],
                save_every_epochs=save_every,
                loo_folder=loo_folder,
                checkpoint_output_dir_base=base_output_dir,
            )
        else:
            llama_lora_faster_train_j.train_model(
                ["", arguments_file],
                save_every_epochs=save_every,
                loo_folder=loo_folder,
                checkpoint_output_dir_base=base_output_dir,
            )
    finally:
        if os.path.exists(arguments_file):
            os.remove(arguments_file)
        try:
            os.rmdir(temp_root)
            os.rmdir(os.path.dirname(temp_root))
        except OSError:
            pass
    
    # Libera esplicitamente la memoria GPU con pulizia aggressiva
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Forza la liberazione su tutti i device
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Pausa breve per dare tempo al sistema di liberare la memoria
    import time
    time.sleep(2)
               
def leave_one_out_trainingV1(model_type, model_name, base_train_set_file_path, base_train_model_output_dir, file_name, epoche, batch_size, save_every=5, gradient_checkpointing=True):
    
    for folder in tqdm(sorted(os.listdir(base_train_set_file_path)), desc="Leave-One-Out Training"):
        print(f"Training model on {folder}")
        folder_path = os.path.join(base_train_set_file_path, folder)
        
        #train_set_file = os.path.join(base_train_set_file_path, 'global_train_text.txt')
        train_set_file = base_train_set_file_path
        
        if os.path.isdir(folder_path):
            dataset_file = os.path.join(folder_path, file_name)
            if os.path.exists(dataset_file):
                
                model_base = base_train_model_output_dir
                print(f"Model base path: {model_base}")
                
                train_model(
                    model_type=model_type,
                    model_name=model_name,
                    max_epochs=epoche,
                    batch_size=batch_size,
                    train_set_file_path=train_set_file,
                    base_output_dir=model_base,
                    loo_folder=folder,
                    save_every=save_every,
                    gradient_checkpointing=gradient_checkpointing
                )
            else:
                print(f"Il file {dataset_file} non esiste.")
