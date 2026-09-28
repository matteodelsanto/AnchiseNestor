#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from functools import partial

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.trainer_callback import TrainerCallback

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import torch

check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
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
    """Callback che salva il modello a intervalli di epoche specifici in cartelle separate"""
    
    def __init__(self, save_every_epochs, output_dir_base, loo_folder, tokenizer):
        self.save_every_epochs = save_every_epochs
        self.output_dir_base = output_dir_base
        self.loo_folder = loo_folder
        self.tokenizer = tokenizer
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Salva il modello ogni `save_every_epochs` epoche"""
        epoch = state.epoch
        if int(epoch) % self.save_every_epochs == 0:
            checkpoint_dir = f"{self.output_dir_base}_{int(epoch)}ep"
            os.makedirs(checkpoint_dir, exist_ok=True)
            if not (self.loo_folder == None):
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

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
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
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

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
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            device_map='auto',
            torch_dtype=torch.float16,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config=config, device_map='auto')

    # Disabilita gradient checkpointing se presente (può causare problemi con LoRA)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    # Abilita i gradienti per l'input embeddings
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        #r=16,
        #lora_alpha=32,
        r=8,
        lora_alpha=16,
        #lora_dropout=0.1,
        lora_dropout=0.15,
        #target_modules=["q_proj", "v_proj"],
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # Stampa i parametri trainabili per debug
    model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        train_dataset = load_dataset_with_skip_logic(data_args.train_file, tokenizer, block_size=1024, loo_folder=loo_folder)

    if training_args.do_eval:
        eval_dataset = load_dataset_with_skip_logic(data_args.validation_file, tokenizer, block_size=1024, loo_folder=loo_folder)

    checkpoint_callback = None
    if training_args.do_train and save_every_epochs != 0:
        save_every = save_every_epochs
        output_base = (checkpoint_output_dir_base or training_args.output_dir).rstrip('/')
        checkpoint_callback = SaveCheckpointCallback(save_every, output_base, loo_folder, tokenizer)

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
    
    trainer.label_names = ["labels"]

    if os.path.exists(training_args.output_dir):
        print(f"Deleting model directory: {training_args.output_dir}")
        for root, dirs, files in os.walk(training_args.output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(training_args.output_dir)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
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
    if training_args.do_eval:
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
        print(f"Deleting model directory: {output_dir_to_clean}")
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
    else:
        print(f"Model directory {output_dir_to_clean} does not exist. Nothing to delete.")
