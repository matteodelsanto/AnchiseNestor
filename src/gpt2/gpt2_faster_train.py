#!/usr/bin/env python
# coding=utf-8

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

from datasets import Dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import check_min_version
import torch

check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(transformers.MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer (backed by tokenizers library)."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login`."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file path."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples to this value if set."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


def load_dataset_with_skip_logic(file_path, tokenizer, block_size, loo_folder=None):
    """
    Load dataset from directory structure, skipping folders as in LazyLanguageModelingDatasetV1.
    Pre-tokenize to fixed block_size to save memory.
    Returns a HuggingFace Dataset with 'input_ids' and 'labels' columns.
    """
    base_dir = os.path.dirname(file_path)
    tokenized_speeches = []

    for root, _, files in os.walk(base_dir):
        folder_name = os.path.basename(root)

        if loo_folder and folder_name == loo_folder:
            print(f"Skipping folder: {folder_name}")
            continue

        for file in files:
            if file == "test_text.txt":
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        tokenized = tokenizer(
                            text,
                            max_length=block_size,
                            truncation=True,
                            padding=False,
                            return_attention_mask=False,
                        )
                        input_ids = tokenized["input_ids"]
                        tokenized_speeches.append({
                            "input_ids": input_ids,
                            "labels": input_ids.copy()
                        })

    return Dataset.from_list(tokenized_speeches)


class SaveCheckpointCallback(TrainerCallback):
    """Callback to save model at specific epoch intervals in separate folders."""
    
    def __init__(self, save_every_epochs, output_dir_base, loo_folder, tokenizer):
        self.save_every_epochs = save_every_epochs
        self.output_dir_base = output_dir_base
        self.loo_folder = loo_folder
        self.tokenizer = tokenizer
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        if int(epoch) % self.save_every_epochs == 0:
            checkpoint_dir = f"{self.output_dir_base}_{int(epoch)}ep"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if self.loo_folder is not None:
                checkpoint_dir = os.path.join(checkpoint_dir, self.loo_folder)
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            kwargs['model'].save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"\nModello salvato all'epoca {int(epoch)} in: {checkpoint_dir}")
        return control


def collate_fn(batch, pad_token_id):
    """Memory-efficient collate function that pads only within batch."""
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
    }


def train_model(args, save_every_epochs=0, loo_folder=None, checkpoint_output_dir_base=None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(args) == 2 and args[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.ERROR)

    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("You must specify either config_name or model_name_or_path.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("You must specify either tokenizer_name or model_name_or_path.")

    tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # Abilita gradient checkpointing se richiesto
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    train_dataset = None
    eval_dataset = None

    if training_args.do_train:
        train_dataset = load_dataset_with_skip_logic(
            data_args.train_file, 
            tokenizer, 
            block_size, 
            loo_folder=loo_folder
        )
        
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))

    if training_args.do_eval:
        eval_dataset = load_dataset_with_skip_logic(
            data_args.validation_file, 
            tokenizer, 
            block_size, 
            loo_folder=loo_folder
        )
        
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))

    checkpoint_callback = None
    if training_args.do_train and save_every_epochs != 0:
        output_base = (checkpoint_output_dir_base or training_args.output_dir).rstrip('/')
        checkpoint_callback = SaveCheckpointCallback(save_every_epochs, output_base, loo_folder, tokenizer)

    from functools import partial
    data_collator = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[checkpoint_callback] if checkpoint_callback else None,
    )

    if os.path.exists(training_args.output_dir):
        print(f"Cleaning model directory: {training_args.output_dir}")
        import shutil
        shutil.rmtree(training_args.output_dir)

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub()

    # Libera esplicitamente la memoria PRIMA di pulire le directory
    # Salva il path prima di eliminare trainer
    output_dir_to_clean = training_args.output_dir
    
    # Sposta il modello su CPU per liberare GPU
    if torch.cuda.is_available():
        model.cpu()
    
    del train_dataset
    del eval_dataset
    del data_collator
    del trainer
    del model
    del tokenizer
    del config
    
    # Pulizia aggressiva della cache GPU per multi-GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Forza la liberazione della cache su tutti i device
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    import gc
    gc.collect()
    
    # Attendi per assicurarti che tutte le risorse siano rilasciate
    import time
    time.sleep(2)
    
    # Forza un'altra garbage collection
    gc.collect()
    
    # Ora pulisci le directory
    if os.path.exists(output_dir_to_clean):
        print(f"Cleaning final model directory: {output_dir_to_clean}")
        import shutil
        # Riprova fino a 3 volte se fallisce
        for attempt in range(3):
            try:
                shutil.rmtree(output_dir_to_clean)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Tentativo {attempt + 1} fallito, riprovo... ({e})")
                    time.sleep(2)
                    gc.collect()
                else:
                    print(f"Impossibile eliminare {output_dir_to_clean}: {e}")
