import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, LlamaForCausalLM
import torch  # Aggiungo l'import di torch per gestire la GPU

# import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import json
import os
from tqdm import tqdm

import time

from utils.perplexity import compute_global_perplexity

import csv

import matplotlib.pyplot as plt

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch.multiprocessing as mp

# Imposta il metodo di avvio a 'spawn' per la compatibilità con CUDA
mp.set_start_method('spawn', force=True)

# Disable logging from libraries
transformers.logging.set_verbosity_error()

import psutil

def get_gpu_memory_usage(device="cuda:0"):
    """Restituisce memoria GPU utilizzata in GB"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device) / (1024**3)
    return 0

def estimate_model_memory(model_type, model_dir, device="cuda:0"):
    """
    Carica temporaneamente il modello e misura la memoria occupata.
    
    Returns:
        memoria in GB
    """
    if model_type == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    elif model_type == 'llama_lora':
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    
    torch.cuda.synchronize(device)
    mem_used = get_gpu_memory_usage(device)
    
    del model
    torch.cuda.empty_cache()
    
    return mem_used

def get_available_gpu_memory(device="cuda:0"):
    """Restituisce memoria GPU disponibile in GB"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        return total - allocated
    return 0

def calculate_max_parallel_models(model_type, model_dir, device="cuda:0", safety_margin=0.1):
    """
    Stima quanti modelli puoi caricare in parallelo su una GPU.
    
    Args:
        model_type: tipo di modello
        model_dir: percorso modello
        device: dispositivo GPU
        safety_margin: margine di sicurezza (0.1 = 10% della memoria totale)
    
    Returns:
        numero massimo di modelli parallelizzabili
    """
    
    # print("Calcolo memoria modello...")
    # print(f"Dispositivo: {device}")
    model_memory = estimate_model_memory(model_type, model_dir, device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    available_memory = total_memory * (1 - safety_margin)
    
    max_models = int(available_memory / model_memory)
    
    # print(f"Modello occupa: {model_memory:.2f} GB")
    # print(f"Memoria totale GPU: {total_memory:.2f} GB")
    # print(f"Memoria disponibile: {available_memory:.2f} GB")
    # print(f"Max modelli paralleli: {max(1, max_models)}")
    
    return max(1, max_models)


def produce_perplexity_values_for_text(model_type, model_dir, text, window=20, output_dir="./", output_file_name="", use_gpu=True, index_to_add=None):
    
    # Verifica se è disponibile una GPU
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    
    if model_type == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    elif model_type == 'llama_lora':
        model = LlamaForCausalLM.from_pretrained(model_dir)
        tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
    
    model = model.to(device)  # Sposta il modello su GPU se disponibile
    
    # Estrai il nome del modello dal percorso
    model_name = os.path.basename(model_dir)
    
    # ----------------------- #
    #       Global PPL        #
    # ----------------------- #
    patient_global_perplexity = compute_global_perplexity(text, model, tokenizer, window, device=device)
    
    # Includi il nome del modello nel nome del file e scrivi solo il valore della perplexity
    if index_to_add is None:
        output_filename = f"{output_dir}{output_file_name}_modello_{model_name}_global_ppl_score.txt"
    else:
        output_filename = f"{output_dir}{output_file_name}_modello_{model_name}_global_ppl_score_{index_to_add}.txt"
    
    with open(output_filename, 'w') as f:
        f.write(f"{patient_global_perplexity}")
    
    # Libera la memoria GPU dopo l'uso
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return patient_global_perplexity

def process_single_folder(args):
    """
    Funzione worker per elaborare una singola cartella.
    
    Args:
        args: una tupla contenente (model_dir, input_folder_path, output_folder_path, 
                                   test_file_name, folder_name, window)
    
    Returns:
        Una tupla (folder_name, perplexity) o (folder_name, error_message)
    """
    model_type, model_dir, input_folder_path, output_folder_path, test_file_name, folder_name, window = args
    
    # Verifica se esiste il file di test nella cartella
    test_file_path = os.path.join(input_folder_path, test_file_name)
    if not os.path.exists(test_file_path):
        return (folder_name, f"File {test_file_name} non trovato")
    
    # Crea la cartella di output corrispondente se non esiste
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Leggi il contenuto del file di test
    try:
        with open(test_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
       
        ppl = produce_perplexity_values_for_text(
            model_type=model_type,
            model_dir=model_dir,
            text=text,
            window=window,
            output_dir=output_folder_path + "/",
            output_file_name=folder_name
        )
        
        return (folder_name, ppl)
    except Exception as e:
        return (folder_name, f"Errore: {str(e)}")

def process_data_folders_parallel(model_type, model_dir, input_base_dir, output_base_dir, test_file_name="test_texts.txt", 
                               window=20, max_workers=None):
    """
    Versione parallela che processa tutti i file di testo nelle cartelle di input
    e salva i risultati nelle cartelle corrispondenti in output.
    
    Args:
        model_dir (str): Percorso della directory contenente il modello GPT-2
        input_base_dir (str): Percorso base della directory di input
        output_base_dir (str): Percorso base della directory di output
        test_file_name (str): Nome del file da cercare in ogni sottocartella
        window (int): Dimensione della finestra per il calcolo della perplexity
        max_workers (int): Numero massimo di worker da utilizzare. Se None, usa il numero di core disponibili
    """
    # Assicurati che la directory di output esista
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Ottieni tutte le sottocartelle nella directory di input
    subfolders = [f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))]
    
    # Prepara gli argomenti per ogni worker
    tasks = []
    for folder in subfolders:
        input_folder_path = os.path.join(input_base_dir, folder)
        output_folder_path = os.path.join(output_base_dir, folder)
        tasks.append((
            model_type,
            model_dir, 
            input_folder_path, 
            output_folder_path, 
            test_file_name, 
            folder, 
            window
        ))
    
    # Se max_workers non è specificato, usa il numero di core disponibili
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Esegui i task in parallelo
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_folder, task) for task in tasks]
        
        # Mostra una barra di progresso
        for future in tqdm(as_completed(futures), total=len(futures), desc="Elaborazione cartelle in parallelo"):
            result = future.result()
            results.append(result)
    
    # Stampa i risultati
    for folder_name, result in sorted(results):
        if isinstance(result, float):
            print(f"Elaborazione completata per {folder_name}, perplexity: {result}")
        else:
            print(f"Problema con {folder_name}: {result}")
    
    return results

# VERSIONE CON GPU

def process_single_folder_gpu(args):
    """
    Funzione worker per elaborare una singola cartella con supporto GPU esplicito.
    
    Args:
        args: una tupla contenente (model_dir, input_folder_path, output_folder_path, 
                                    test_file_name, folder_name, window, gpu_id)
    
    Returns:
        Una tupla (folder_name, perplexity) o (folder_name, error_message)
    """
    model_type, model_dir, input_folder_path, output_folder_path, test_file_name, folder_name, window, gpu_id = args
    
    # Imposta la GPU specifica per questo processo
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Verifica se esiste il file di test nella cartella
    test_file_path = os.path.join(input_folder_path, test_file_name)
    if not os.path.exists(test_file_path):
        return (folder_name, f"File {test_file_name} non trovato")
    
    # Crea la cartella di output corrispondente se non esiste
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Leggi il contenuto del file di test
    try:
        with open(test_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        ppl = produce_perplexity_values_for_text(
            model_type=model_type,
            model_dir=model_dir,
            text=text,
            window=window,
            output_dir=output_folder_path + "/",
            output_file_name=folder_name,
            use_gpu=True  # Forza l'uso della GPU
        )
        
        return (folder_name, ppl)
    except Exception as e:
        return (folder_name, f"Errore: {str(e)}")

def process_data_folders_multi_gpu(model_type, model_dir, input_base_dir, output_base_dir, test_file_name="test_texts.txt", 
                                   window=20, num_gpus=None, fake_leave_one_out=False, leap=0):
    """
    Versione parallela che utilizza multiple GPU per accelerare l'elaborazione.
    
    Args:
        model_dir (str): Percorso della directory contenente il modello GPT-2
        input_base_dir (str): Percorso base della directory di input
        output_base_dir (str): Percorso base della directory di output
        test_file_name (str): Nome del file da cercare in ogni sottocartella
        window (int): Dimensione della finestra per il calcolo della perplexity
        num_gpus (int): Numero di GPU da utilizzare. Se None, usa tutte le GPU disponibili.
    """
    # Verifica quante GPU sono disponibili
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        print("ATTENZIONE: Nessuna GPU disponibile. Verrà utilizzata la CPU.")
        return process_data_folders_parallel(model_type, model_dir, input_base_dir, output_base_dir, 
                                           test_file_name, window, max_workers=None, leap=leap)
    
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    print(f"Utilizzo {num_gpus} GPU per l'elaborazione parallela.")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Assicurati che la directory di output esista
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Ottieni tutte le sottocartelle nella directory di input
    subfolders = [f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))]
    
    # Dividi il lavoro tra le GPU
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, folder in enumerate(subfolders):
        gpu_id = i % num_gpus
        gpu_tasks[gpu_id].append(folder)
    
    # Crea processi per ciascuna GPU
    processes = []
    result_queue = mp.Queue()
    
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:  # Salta le GPU senza task assegnati
            continue
            
        # Crea un processo per questa GPU
        p = mp.Process(
            target=_process_gpu_batch,
            args=(
                gpu_id,
                model_type,
                model_dir,
                input_base_dir,
                output_base_dir,
                test_file_name,
                gpu_tasks[gpu_id],
                window,
                result_queue,
                fake_leave_one_out,
                leap
            )
        )
        processes.append(p)
        p.start()
    
    # Raccogliere i risultati
    results = []
    with tqdm(total=len(subfolders), desc=f"Elaborazione parallela su {num_gpus} GPU") as pbar:
        for _ in range(len(subfolders)):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)
    
    # Attendere che tutti i processi terminino
    for p in processes:
        p.join()
    
    # Stampa i risultati
    for folder_name, result in sorted(results):
        if isinstance(result, float):
            print(f"Elaborazione completata per {folder_name}, perplexity: {result}")
        else:
            print(f"Problema con {folder_name}: {result}")
    
    return results

def _process_folder_on_gpu(folder, input_base_dir, output_base_dir, test_file_name, model, tokenizer, device, model_name, fake_leave_one_out, leap, window, result_queue):
    """Processa una singola cartella su GPU."""
    input_folder_path = os.path.join(input_base_dir, folder)
    output_folder_path = os.path.join(output_base_dir, folder)
    
    test_file_prefix = test_file_name.split(".")[0]
    matching_files = [f for f in os.listdir(input_folder_path) if f.startswith(test_file_prefix)]
    
    if not matching_files:
        matching_files = [test_file_name] if os.path.exists(os.path.join(input_folder_path, test_file_name)) else []
    
    if not matching_files:
        result_queue.put((folder, f"File {test_file_name} non trovato"))
        return
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Caso: file singolo
    if len(matching_files) == 1 and matching_files[0] == test_file_name:
        test_file_path = os.path.join(input_folder_path, test_file_name)
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            ppl = compute_global_perplexity(text, model, tokenizer, window_size=window, device=device, leap=leap)
            
            suffix = "_leave_one_out_ppl_score.txt" if fake_leave_one_out else "_global_ppl_score.txt"
            output_filename = f"{output_folder_path}/{folder}_modello_{model_name}{suffix}"
            
            with open(output_filename, 'w') as f:
                f.write(f"{ppl}")
            
            result_queue.put((folder, ppl))
        except Exception as e:
            result_queue.put((folder, f"Errore: {str(e)}"))
    
    # Caso: file multipli
    else:
        for i, m in enumerate(matching_files):
            test_file_path = os.path.join(input_folder_path, m)
            if not os.path.exists(test_file_path):
                continue
            
            try:
                with open(test_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                ppl = compute_global_perplexity(text, model, tokenizer, window=20, device=device, leap=leap)
                
                suffix = "_leave_one_out_ppl_score" if fake_leave_one_out else "_global_ppl_score"
                output_filename = f"{output_folder_path}/{folder}_modello_{model_name}{suffix}_{i}.txt"
                
                with open(output_filename, 'w') as f:
                    f.write(f"{ppl}")
                
                result_queue.put((folder, ppl))
            except Exception as e:
                result_queue.put((folder, f"Errore: {str(e)}"))

def _process_gpu_batch(gpu_id, model_type, model_dir, input_base_dir, output_base_dir, test_file_name, folders, window, result_queue, fake_leave_one_out=False, leap=0):
    """Elabora un batch di cartelle su una GPU specifica con più modelli in memoria."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"{device}:{gpu_id}"
    
    # Stima quanti modelli puoi caricare
    max_models = calculate_max_parallel_models(model_type, model_dir, device, safety_margin=0.15)
    max_models = min(max_models, 2)  # Limita a massimo 10 modelli
    # print(f"GPU {gpu_id}: caricando {max_models} modello(i) in memoria")
    
    # Carica più istanze del modello
    models = []
    tokenizers = []
    for i in range(max_models):
        if model_type == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        elif model_type == 'llama_lora':
            model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
            tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
        
        models.append(model)
        tokenizers.append(tokenizer)
    
    model_name = os.path.basename(model_dir)
    
    
    # Processa le cartelle usando i modelli in pool
    with ThreadPoolExecutor(max_workers=max_models) as executor:
        futures = []
        for idx, folder in enumerate(folders):
            model_idx = idx % max_models  # Assegna a turno un modello
            future = executor.submit(
                _process_folder_on_gpu,
                folder, input_base_dir, output_base_dir, test_file_name,
                models[model_idx], tokenizers[model_idx], device, model_name, 
                fake_leave_one_out, leap, window, result_queue
            )
            futures.append(future)
        
        # Attendi completamento
        for future in futures:
            future.result()
    
    # Libera memoria GPU
    for model in models:
        del model
    if device == "cuda":
        torch.cuda.empty_cache()

####### SEZIONE per gestione del leave one out #######

def process_leave_one_out_single_gpu(gpu_id, patient_folders, input_base_dir, model_type, models_base_dir, output_base_dir, 
                                    test_file_name, window, result_queue, leap = 0):
    """
    Funzione helper che elabora un batch di cartelle su una specifica GPU,
    con un modello differente per ogni paziente (leave-one-out).
    
    Args:
        gpu_id: ID della GPU da utilizzare
        patient_folders: Lista di cartelle paziente da elaborare
        input_base_dir: Directory base di input
        models_base_dir: Directory base contenente i modelli (uno per paziente)
        output_base_dir: Directory base di output
        test_file_name: Nome del file da elaborare
        window: Dimensione della finestra
        result_queue: Coda per restituire i risultati
    """
    # Imposta la GPU specifica per questo processo
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = device + f":{gpu_id}"  
    
    for patient_folder in patient_folders:
        # Il modello ha lo stesso nome della cartella del paziente
        model_path = os.path.join(models_base_dir, patient_folder)
        input_folder_path = os.path.join(input_base_dir, patient_folder)
        output_folder_path = os.path.join(output_base_dir, patient_folder)
        
        test_file_prefix = test_file_name.split(".")[0]
        #test_file_prefix = "_".join(test_file_prefix.split("_")[:-1])  # Rimuovi l'ultima parte dopo l'underscore 
        #test_file_prefix = "test_text"
        
        # Loop through the folder and save the list of file names that start with the prefix
        matching_files = []
        for file in os.listdir(input_folder_path):
            if file.startswith(test_file_prefix):
                matching_files.append(file)

        # If no matching files found, use the original test_file_name
        if not matching_files and os.path.exists(test_file_path):
            matching_files = [test_file_name]
        
        # # Verifica se esiste il file di test nella cartella
        # test_file_path = os.path.join(input_folder_path, test_file_name)
        # if not os.path.exists(test_file_path):
        #     result_queue.put((patient_folder, f"File {test_file_name} non trovato"))
        #     continue
            
        # Verifica se esiste il modello per questo paziente
        if not os.path.exists(model_path):
            result_queue.put((patient_folder, f"Modello per {patient_folder} non trovato in {model_path}"))
            continue
        
        # Crea la cartella di output se non esiste
        os.makedirs(output_folder_path, exist_ok=True)
        
        try:
            # Carica il modello specifico per questo paziente
            if model_type == 'gpt2':
                model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
                tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
            elif model_type == 'llama_lora':
                model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
                tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
            
            for i,m in enumerate(matching_files):
                test_file_path = os.path.join(input_folder_path, m)
            
                # Leggi il testo del paziente
                with open(test_file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                # Calcola la perplexity
                ppl = compute_global_perplexity(text, model, tokenizer, window, device=device, leap=leap)
                
                #estrai da models_base_dir il nome del modello
                model_name = os.path.basename(models_base_dir)
                # Salva il risultato
                if len(matching_files) == 1:
                    output_filename = f"{output_folder_path}/{patient_folder}_modello_{model_name}_leave_one_out_ppl_score.txt"
                else:
                    output_filename = f"{output_folder_path}/{patient_folder}_modello_{model_name}_leave_one_out_ppl_score_{i}.txt"
                with open(output_filename, 'w') as f:
                    f.write(f"{ppl}")
                
                result_queue.put((patient_folder, ppl))
            
            # Libera la memoria GPU dopo ogni paziente
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            result_queue.put((patient_folder, f"Errore: {str(e)}"))

def process_leave_one_out_multi_gpu(model_type, models_base_dir, input_base_dir, output_base_dir, test_file_name="test_text.txt", window=20, num_gpus=None, leap = 0):
    """
    Gestisce l'elaborazione leave-one-out distribuendo il carico su multiple GPU.
    Per ogni paziente, carica un modello specifico con lo stesso nome della cartella paziente.
    
    Args:
        models_base_dir (str): Directory base contenente i modelli (uno per paziente)
        input_base_dir (str): Percorso base della directory di input
        output_base_dir (str): Percorso base della directory di output
        test_file_name (str): Nome del file da cercare in ogni sottocartella paziente
        window (int): Dimensione della finestra per il calcolo della perplexity
        num_gpus (int): Numero di GPU da utilizzare. Se None, usa tutte le GPU disponibili.
    """
    # Verifica quante GPU sono disponibili
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("ATTENZIONE: Nessuna GPU disponibile. Impossibile procedere con l'elaborazione leave-one-out.")
        return []
    
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    print(f"Utilizzo {num_gpus} GPU per l'elaborazione leave-one-out.")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Assicurati che la directory di output esista
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Ottieni tutte le sottocartelle paziente nella directory di input
    patient_folders = [f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))]
    
    # Dividi il lavoro tra le GPU
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, folder in enumerate(patient_folders):
        gpu_id = i % num_gpus
        gpu_tasks[gpu_id].append(folder)
    
    # Crea processi per ciascuna GPU
    processes = []
    result_queue = mp.Queue()
    
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:  # Salta le GPU senza task assegnati
            continue
            
        # Crea un processo per questa GPU
        p = mp.Process(
            target=process_leave_one_out_single_gpu,
            args=(
                gpu_id,
                gpu_tasks[gpu_id],
                input_base_dir,
                model_type,
                models_base_dir,
                output_base_dir,
                test_file_name,
                window,
                result_queue,
                leap
            )
        )
        processes.append(p)
        p.start()
    
    # Raccogliere i risultati
    results = []
    with tqdm(total=len(patient_folders), desc=f"Elaborazione leave-one-out su {num_gpus} GPU") as pbar:
        for _ in range(len(patient_folders)):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)
    
    # Attendere che tutti i processi terminino
    for p in processes:
        p.join()
    
    # Stampa i risultati
    for folder_name, result in sorted(results):
        if isinstance(result, float):
            print(f"Elaborazione leave-one-out completata per {folder_name}, perplexity: {result}")
        else:
            print(f"Problema con {folder_name}: {result}")
    
    return results