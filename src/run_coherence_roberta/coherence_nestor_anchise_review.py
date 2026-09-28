import argparse
import json
import os
import random
import shutil
import time
import inspect
import datetime
from pprint import pformat

import pandas as pd
import torch
from datasets import Dataset
from torch import nn
from tqdm import tqdm
from transformers import (
    EarlyStoppingCallback,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)


SEED = 0
NEGATIVE_TO_POSITIVE_RATIO = 8
DEV_RATIO = 0.1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

NESTOR_RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "resources", "data", "input", "coherence", "nestor", "corpus_elderly.csv")
ANCHISE_PATIENTS_DIR = os.path.join(PROJECT_ROOT, "resources", "data", "input", "coherence", "anchise")

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "coherence", "nestor_anchise_review_runs")
GLOBAL_MODEL_SUBJECT_ID = "global_nestor"


def get_latest_run_id(results_root=RESULTS_ROOT):
    if not os.path.exists(results_root):
        return None

    run_ids = []
    for entry in os.listdir(results_root):
        entry_path = os.path.join(results_root, entry)
        if os.path.isdir(entry_path):
            run_ids.append(entry)

    if len(run_ids) == 0:
        return None

    return sorted(run_ids)[-1]


def resolve_run_id(results_root=RESULTS_ROOT):
    requested_run_id = os.environ.get("COHERENCE_NESTOR_ANCHISE_REVIEW_RUN_ID")
    if requested_run_id is not None and requested_run_id.strip() != "":
        return requested_run_id.strip()

    use_latest_run = os.environ.get("COHERENCE_NESTOR_ANCHISE_REVIEW_USE_LATEST_RUN", "").strip().lower()
    if use_latest_run in {"1", "true", "yes"}:
        latest_run_id = get_latest_run_id(results_root=results_root)
        if latest_run_id is None:
            raise ValueError("Nessun run precedente trovato da riutilizzare.")
        return latest_run_id

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


RUN_ID = resolve_run_id()
RUN_ROOT = os.path.join(RESULTS_ROOT, RUN_ID)
PREPROCESSED_NESTOR_TOUSE_DIR = os.path.join(RUN_ROOT, "preprocessed", "nestor_touse")
NESTOR_LOSO_DIR = os.path.join(RUN_ROOT, "loso_splits")
MODELS_DIR = os.path.join(RUN_ROOT, "models")
TRAINING_RUNS_DIR = os.path.join(RUN_ROOT, "training_runs")
LOGGING_DIR = os.path.join(RUN_ROOT, "logs")
GLOBAL_ANCHISE_RESULTS_DIR = os.path.join(RUN_ROOT, "global_anchise_eval")
RUN_STATE_PATH = os.path.join(RUN_ROOT, "run_state.json")


def ensure_run_directories():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)
    os.makedirs(PREPROCESSED_NESTOR_TOUSE_DIR, exist_ok=True)
    os.makedirs(NESTOR_LOSO_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TRAINING_RUNS_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    os.makedirs(GLOBAL_ANCHISE_RESULTS_DIR, exist_ok=True)


def default_run_state():
    return {
        "preprocessing": {"status": "pending"},
        "global": {"status": "pending"},
        "loso": {"completed_subjects": [], "current_subject": None},
        "final_metrics": {"status": "pending"},
    }


def load_run_state():
    ensure_run_directories()
    if not os.path.exists(RUN_STATE_PATH):
        return default_run_state()

    with open(RUN_STATE_PATH, "r") as f:
        return json.load(f)


def save_run_state(state):
    ensure_run_directories()
    with open(RUN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def harmonic_mean(values):
    non_zero_values = [value for value in values if value > 0]
    if len(non_zero_values) != len(values) or len(values) == 0:
        return 0.0

    denominator = sum(1 / value for value in values)
    if denominator == 0:
        return 0.0

    return len(values) / denominator


def read_average_coherence(result_file_path):
    with open(result_file_path, "r") as f:
        return float(f.read().split(" ")[-1])


def update_confusion_counts(confusion_counts, gold_label, predicted_label):
    if gold_label == "dementia":
        if predicted_label == "dementia":
            confusion_counts["dementia"]["tp"] += 1
            confusion_counts["control"]["tn"] += 1
        else:
            confusion_counts["dementia"]["fn"] += 1
            confusion_counts["control"]["fp"] += 1
    else:
        if predicted_label == "control":
            confusion_counts["control"]["tp"] += 1
            confusion_counts["dementia"]["tn"] += 1
        else:
            confusion_counts["control"]["fn"] += 1
            confusion_counts["dementia"]["fp"] += 1


def compute_binary_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def collect_predictions_from_results(results_dir, gold_label, confusion_counts):
    if not os.path.exists(results_dir):
        print(f"Directory risultati non trovata per {gold_label}: {results_dir}")
        return 0

    processed_sessions = 0
    for subject_id in sorted(os.listdir(results_dir)):
        subject_path = os.path.join(results_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        test_dir = os.path.join(subject_path, "test")
        if not os.path.exists(test_dir):
            continue

        for session in sorted(os.listdir(test_dir)):
            session_path = os.path.join(test_dir, session)
            if not os.path.isdir(session_path):
                continue

            coherent_result = os.path.join(session_path, "avg_coherence_of_coherent_couples.txt")
            incoherent_result = os.path.join(session_path, "avg_coherence_of_incoherent_couples.txt")
            if not os.path.exists(coherent_result) or not os.path.exists(incoherent_result):
                continue

            adj_mean = read_average_coherence(coherent_result)
            nadj_mean = read_average_coherence(incoherent_result)
            predicted_label = "control" if adj_mean > nadj_mean else "dementia"
            update_confusion_counts(confusion_counts, gold_label, predicted_label)
            processed_sessions += 1

    return processed_sessions


def collect_predictions_from_flat_results(results_dir, gold_label, confusion_counts):
    if not os.path.exists(results_dir):
        print(f"Directory risultati non trovata per {gold_label}: {results_dir}")
        return 0

    processed_sessions = 0
    for session in sorted(os.listdir(results_dir)):
        session_path = os.path.join(results_dir, session)
        if not os.path.isdir(session_path):
            continue

        coherent_result = os.path.join(session_path, "avg_coherence_of_coherent_couples.txt")
        incoherent_result = os.path.join(session_path, "avg_coherence_of_incoherent_couples.txt")
        if not os.path.exists(coherent_result) or not os.path.exists(incoherent_result):
            continue

        adj_mean = read_average_coherence(coherent_result)
        nadj_mean = read_average_coherence(incoherent_result)
        predicted_label = "control" if adj_mean > nadj_mean else "dementia"
        update_confusion_counts(confusion_counts, gold_label, predicted_label)
        processed_sessions += 1

    return processed_sessions


def find_missing_control_results(loso_root=NESTOR_LOSO_DIR, touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR):
    missing_results = []
    sessions_by_subject = list_sessions_by_subject(touse_dir=touse_dir)

    for subject_id, sessions in sorted(sessions_by_subject.items()):
        for session in sessions:
            session_path = os.path.join(loso_root, subject_id, "test", session)
            coherent_result = os.path.join(session_path, "avg_coherence_of_coherent_couples.txt")
            incoherent_result = os.path.join(session_path, "avg_coherence_of_incoherent_couples.txt")
            if not os.path.exists(coherent_result) or not os.path.exists(incoherent_result):
                missing_results.append({"subject_id": subject_id, "session": session})

    return missing_results


class RobertaForCoherenceClassification(nn.Module):
    def __init__(self, num_labels=1):
        super(RobertaForCoherenceClassification, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.roberta.config.hidden_size, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output).squeeze()

        if labels is not None:
            loss_fct = nn.BCELoss()
            logits = logits.view(-1)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels.float())
            return loss, logits
        return logits


def tokenize_function(data):
    return tokenizer(
        data["U1"],
        data["U2"],
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )


def get_subject_id_from_session(session_id):
    return str(session_id)


def get_session_id_from_file_name(file_name):
    if file_name.endswith("_adj.txt"):
        return file_name[:-8]
    if file_name.endswith("_nadj.txt"):
        return file_name[:-9]
    return file_name


def normalize_speaker_label(value):
    return str(value).strip().upper()


def sanitize_utterance(text):
    if text is None:
        return ""
    return str(text).strip()


def build_pairs_from_utterances(utterances):
    adjacent_pairs = []
    non_adjacent_pairs = []

    for i, current_utterance in enumerate(utterances):
        if current_utterance == "":
            continue

        if i != len(utterances) - 1:
            next_utterance = utterances[i + 1]
            if next_utterance != "":
                adjacent_pairs.append((current_utterance, next_utterance))

        for j in range(i + 2, len(utterances)):
            future_utterance = utterances[j]
            if future_utterance != "":
                non_adjacent_pairs.append((current_utterance, future_utterance))

    return adjacent_pairs, non_adjacent_pairs


def write_pairs_to_txt(output_path, pairs):
    with open(output_path, "w") as f:
        for u1, u2 in pairs:
            if u1 != "" and u2 != "":
                f.write(f"{u1}\t{u2}\n")


def preprocess_nestor_corpus(raw_csv_path=NESTOR_RAW_CSV_PATH, output_dir=PREPROCESSED_NESTOR_TOUSE_DIR):
    ensure_run_directories()
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(raw_csv_path)
    required_columns = {"NUM BATTUTA", "PARLANTE", "TRASCRIZIONE CONVERSAZIONI", "NUM SEDUTA"}
    missing_columns = required_columns.difference(df.columns)
    if len(missing_columns) > 0:
        raise ValueError(f"Colonne mancanti in corpus_elderly.csv: {sorted(missing_columns)}")

    patient_df = df[df["PARLANTE"].map(normalize_speaker_label) == "A"].copy()
    patient_df["TRASCRIZIONE CONVERSAZIONI"] = patient_df["TRASCRIZIONE CONVERSAZIONI"].map(sanitize_utterance)
    patient_df["NUM SEDUTA"] = patient_df["NUM SEDUTA"].astype(str).str.strip()
    patient_df["NUM BATTUTA"] = pd.to_numeric(patient_df["NUM BATTUTA"], errors="coerce")
    patient_df = patient_df.dropna(subset=["NUM BATTUTA"])

    for session_id, session_df in patient_df.groupby("NUM SEDUTA"):
        session_df = session_df.sort_values("NUM BATTUTA")
        utterances = [text for text in session_df["TRASCRIZIONE CONVERSAZIONI"].tolist() if text != ""]
        adjacent_pairs, non_adjacent_pairs = build_pairs_from_utterances(utterances)

        if len(adjacent_pairs) == 0 or len(non_adjacent_pairs) == 0:
            continue

        write_pairs_to_txt(os.path.join(output_dir, f"{session_id}_adj.txt"), adjacent_pairs)
        write_pairs_to_txt(os.path.join(output_dir, f"{session_id}_nadj.txt"), non_adjacent_pairs)


def list_sessions_by_subject(touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR):
    sessions_by_subject = {}

    for file_name in os.listdir(touse_dir):
        if not file_name.endswith("_adj.txt"):
            continue

        session_id = get_session_id_from_file_name(file_name)
        nadj_file = f"{session_id}_nadj.txt"
        if not os.path.exists(os.path.join(touse_dir, nadj_file)):
            continue

        subject_id = get_subject_id_from_session(session_id)
        sessions_by_subject.setdefault(subject_id, []).append(session_id)

    for subject_id in sessions_by_subject:
        sessions_by_subject[subject_id] = sorted(sessions_by_subject[subject_id])

    return sessions_by_subject


def read_pairs_from_txt(file_path):
    pairs = []
    if not os.path.exists(file_path):
        return pairs

    with open(file_path, "r") as f:
        lines = f.read().split("\n")

    for line in lines:
        if line == "" or line == " ":
            continue

        split_line = line.split("\t")
        if len(split_line) < 2:
            continue

        u1 = split_line[0].strip()
        u2 = split_line[1].strip()
        if u1 == "" or u2 == "":
            continue
        pairs.append((u1, u2))

    return pairs


def build_training_dataframe_from_sessions(train_sessions, touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR):
    all_adj = []
    all_nadj = []

    for session_id in train_sessions:
        adj_path = os.path.join(touse_dir, f"{session_id}_adj.txt")
        nadj_path = os.path.join(touse_dir, f"{session_id}_nadj.txt")

        adj_pairs = read_pairs_from_txt(adj_path)
        nadj_pairs = read_pairs_from_txt(nadj_path)

        all_adj.extend(adj_pairs)
        all_nadj.extend(nadj_pairs)

    random_generator = random.Random(SEED)
    random_generator.shuffle(all_nadj)
    all_nadj = all_nadj[: len(all_adj) * NEGATIVE_TO_POSITIVE_RATIO]

    data_rows = []
    for u1, u2 in all_adj:
        data_rows.append([u1, u2, 1])
    for u1, u2 in all_nadj:
        data_rows.append([u1, u2, 0])

    random_generator.shuffle(data_rows)
    return pd.DataFrame(data_rows, columns=["U1", "U2", "label"])


def build_training_dataframe_from_all_available_sessions(touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR):
    sessions_by_subject = list_sessions_by_subject(touse_dir=touse_dir)
    all_sessions = []
    for subject_id in sorted(sessions_by_subject.keys()):
        all_sessions.extend(sessions_by_subject[subject_id])
    return build_training_dataframe_from_sessions(all_sessions, touse_dir=touse_dir)


def materialize_loso_split(subject_id, sessions_by_subject, touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR, loso_root=NESTOR_LOSO_DIR):
    all_subjects = sorted(sessions_by_subject.keys())
    test_sessions = sessions_by_subject[subject_id]
    train_sessions = []

    for sid in all_subjects:
        if sid == subject_id:
            continue
        train_sessions.extend(sessions_by_subject[sid])

    split_dir = os.path.join(loso_root, subject_id)
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_df = build_training_dataframe_from_sessions(train_sessions, touse_dir=touse_dir)
    train_csv_path = os.path.join(train_dir, "all_utterances.csv")
    train_df.to_csv(train_csv_path, index=False)

    for session_id in test_sessions:
        session_test_dir = os.path.join(test_dir, session_id)
        os.makedirs(session_test_dir, exist_ok=True)

        src_adj = os.path.join(touse_dir, f"{session_id}_adj.txt")
        src_nadj = os.path.join(touse_dir, f"{session_id}_nadj.txt")

        dst_adj = os.path.join(session_test_dir, f"{session_id}.txt_adj.txt")
        dst_nadj = os.path.join(session_test_dir, f"{session_id}.txt_nadj.txt")

        if os.path.exists(src_adj):
            shutil.copyfile(src_adj, dst_adj)
        if os.path.exists(src_nadj):
            shutil.copyfile(src_nadj, dst_nadj)

    return {
        "subject_id": subject_id,
        "split_dir": split_dir,
        "train_csv_path": train_csv_path,
        "test_dir": test_dir,
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
    }


def sanitize_pair_dataframe(dataframe):
    if len(dataframe) == 0:
        return dataframe.copy()

    sanitized = dataframe.copy()
    for column in ["U1", "U2"]:
        if column not in sanitized.columns:
            raise KeyError(f"Colonna mancante nel dataframe delle coppie: {column}")

        sanitized[column] = sanitized[column].fillna("").astype(str).str.strip()

    valid_rows = (sanitized["U1"] != "") & (sanitized["U2"] != "")
    return sanitized.loc[valid_rows].reset_index(drop=True)


def tokenize_dataframe(dataframe):
    dataframe = sanitize_pair_dataframe(dataframe)
    data_dict = dataframe.to_dict(orient="records")

    for dictionary in data_dict:
        dictionary["U1"] = str(dictionary["U1"])
        dictionary["U2"] = str(dictionary["U2"])

    dataset = Dataset.from_list(data_dict)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["U1", "U2"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset


def split_train_dev_dataframe(dataframe, dev_ratio=DEV_RATIO):
    random_generator = random.Random(SEED)

    train_parts = []
    dev_parts = []
    for _, label_df in dataframe.groupby("label"):
        label_df = label_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        if len(label_df) <= 1:
            train_parts.append(label_df)
            continue

        dev_size = int(round(len(label_df) * dev_ratio))
        dev_size = max(1, dev_size)
        dev_size = min(dev_size, len(label_df) - 1)

        dev_parts.append(label_df.iloc[:dev_size])
        train_parts.append(label_df.iloc[dev_size:])

    train_df = pd.concat(train_parts, ignore_index=True)
    dev_df = pd.concat(dev_parts, ignore_index=True) if len(dev_parts) > 0 else pd.DataFrame(columns=dataframe.columns)

    train_df = train_df.sample(frac=1, random_state=random_generator.randint(0, 10**9)).reset_index(drop=True)
    if len(dev_df) > 0:
        dev_df = dev_df.sample(frac=1, random_state=random_generator.randint(0, 10**9)).reset_index(drop=True)

    return train_df, dev_df


def prepare_data_from_csv(data_csv_path):
    data = pd.read_csv(data_csv_path)
    data = sanitize_pair_dataframe(data)
    train_df, dev_df = split_train_dev_dataframe(data)

    train_dataset = tokenize_dataframe(train_df)
    dev_dataset = tokenize_dataframe(dev_df) if len(dev_df) > 0 else None
    return train_dataset, dev_dataset


def build_training_arguments(output_dir):
    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": 50,
    }

    if "per_device_train_batch_size" in supported:
        kwargs["per_device_train_batch_size"] = 128
    elif "per_gpu_train_batch_size" in supported:
        kwargs["per_gpu_train_batch_size"] = 128

    if "logging_dir" in supported:
        kwargs["logging_dir"] = LOGGING_DIR

    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluate_during_training" in supported:
        kwargs["evaluate_during_training"] = True

    if "save_strategy" in supported:
        kwargs["save_strategy"] = "epoch"

    if "load_best_model_at_end" in supported:
        kwargs["load_best_model_at_end"] = True

    if "metric_for_best_model" in supported:
        kwargs["metric_for_best_model"] = "eval_loss"

    if "greater_is_better" in supported:
        kwargs["greater_is_better"] = False

    if "save_total_limit" in supported:
        kwargs["save_total_limit"] = 1

    if "do_eval" in supported:
        kwargs["do_eval"] = True

    return TrainingArguments(**kwargs)


def model_path_for_subject(subject_id):
    return os.path.join(MODELS_DIR, f"roBERTa_coherence_nestor_subject_{subject_id}")


def training_output_dir_for_subject(subject_id):
    return os.path.join(TRAINING_RUNS_DIR, f"subject_{subject_id}")


def remove_trainer_checkpoints(training_output_dir):
    if not os.path.exists(training_output_dir):
        return

    for entry in os.listdir(training_output_dir):
        entry_path = os.path.join(training_output_dir, entry)
        if entry.startswith("checkpoint-") and os.path.isdir(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)


def cleanup_model_artifacts(subject_id):
    model_path = model_path_for_subject(subject_id)
    if os.path.exists(model_path):
        os.remove(model_path)

    remove_trainer_checkpoints(training_output_dir_for_subject(subject_id))


def cleanup_global_eval_results():
    if not os.path.exists(GLOBAL_ANCHISE_RESULTS_DIR):
        return

    for entry in os.listdir(GLOBAL_ANCHISE_RESULTS_DIR):
        entry_path = os.path.join(GLOBAL_ANCHISE_RESULTS_DIR, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)
        else:
            os.remove(entry_path)


def train_model(train_dataset, dev_dataset, subject_id):
    ensure_run_directories()
    model = RobertaForCoherenceClassification()
    starting_time = time.time()

    training_on = train_dataset.num_rows
    validating_on = dev_dataset.num_rows if dev_dataset is not None else 0

    output_dir = training_output_dir_for_subject(subject_id)
    os.makedirs(output_dir, exist_ok=True)

    training_args = build_training_arguments(output_dir)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (logits > 0.5).astype(int)
        accuracy = (preds == labels).mean()
        with open(os.path.join(output_dir, f"metrics_subject_{subject_id}.txt"), "a") as f:
            f.write(f"Accuracy: {accuracy}\n")
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    trainer.train()

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = model_path_for_subject(subject_id)
    torch.save(model.state_dict(), model_path)

    final_time = time.time()
    total_elapsed_time = final_time - starting_time
    with open(os.path.join(output_dir, f"metrics_subject_{subject_id}.txt"), "a") as f:
        f.write(f"Starting time: {starting_time}\n")
        f.write(f"Training on {training_on} samples\n")
        f.write(f"Validating on {validating_on} samples\n")
        f.write(f"Final time: {final_time}\n")
        f.write(f"Total elapsed time: {total_elapsed_time}\n")

    return model_path


def train_global_model(touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR):
    ensure_run_directories()
    cleanup_model_artifacts(GLOBAL_MODEL_SUBJECT_ID)

    global_output_dir = training_output_dir_for_subject(GLOBAL_MODEL_SUBJECT_ID)
    os.makedirs(global_output_dir, exist_ok=True)

    train_df = build_training_dataframe_from_all_available_sessions(touse_dir=touse_dir)
    train_csv_path = os.path.join(global_output_dir, "all_utterances.csv")
    train_df.to_csv(train_csv_path, index=False)

    train_dataset, dev_dataset = prepare_data_from_csv(train_csv_path)
    print("ALLENAMENTO MODELLO GLOBALE NESTOR")
    return train_model(train_dataset, dev_dataset, GLOBAL_MODEL_SUBJECT_ID)


def compute_coherence(data_path, model_path):
    model = RobertaForCoherenceClassification()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    test_data = pd.read_csv(data_path, sep="\t")
    test_data_dict = test_data.to_dict(orient="records")

    if len(test_data_dict) == 0:
        return 0.0

    testset = tokenizer(
        [pair["U1"] for pair in test_data_dict],
        [pair["U2"] for pair in test_data_dict],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128,
    )

    with torch.no_grad():
        input_ids = testset["input_ids"]
        attention_mask = testset["attention_mask"]
        outputs = model(input_ids, attention_mask)

    logits = outputs.cpu().numpy()
    return logits.mean()


def write_tsv_from_txt_pairs(input_txt_path, output_tsv_path, label, session_id):
    with open(output_tsv_path, "w") as fw:
        fw.write("U1\tU2\tlabel\tsession\n")
        with open(input_txt_path) as fr:
            lines = fr.read().split("\n")
            for u in lines:
                if u != "" and u != " ":
                    split_line = u.split("\t")
                    if len(split_line) >= 2 and split_line[0] != "" and split_line[1] != "":
                        fw.write(f"{split_line[0]}\t{split_line[1]}\t{label}\t{session_id}\n")


def analyze_pairs_session(session_id, coherent_txt_path, incoherent_txt_path, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)

    coherent_data_path = os.path.join(output_dir, f"{session_id}.txt_adj.tsv")
    incoherent_data_path = os.path.join(output_dir, f"{session_id}.txt_nadj.tsv")

    write_tsv_from_txt_pairs(coherent_txt_path, coherent_data_path, 1, session_id)
    write_tsv_from_txt_pairs(incoherent_txt_path, incoherent_data_path, 0, session_id)

    coherent_coherence = compute_coherence(coherent_data_path, model_path)
    incoherent_coherence = compute_coherence(incoherent_data_path, model_path)

    with open(os.path.join(output_dir, "avg_coherence_of_coherent_couples.txt"), "w") as f:
        f.write(f"AVG coherence: {coherent_coherence}\n")
    with open(os.path.join(output_dir, "avg_coherence_of_incoherent_couples.txt"), "w") as f:
        f.write(f"AVG coherence: {incoherent_coherence}")


def analyze_coherence_on_subject_sessions(subject_test_dir, model_path):
    for session in tqdm(sorted(os.listdir(subject_test_dir)), desc="Session"):
        session_path = os.path.join(subject_test_dir, session)
        if not os.path.isdir(session_path):
            continue

        coherent_txt_path = os.path.join(session_path, f"{session}.txt_adj.txt")
        incoherent_txt_path = os.path.join(session_path, f"{session}.txt_nadj.txt")
        if not os.path.exists(coherent_txt_path) or not os.path.exists(incoherent_txt_path):
            continue

        analyze_pairs_session(
            session_id=session,
            coherent_txt_path=coherent_txt_path,
            incoherent_txt_path=incoherent_txt_path,
            output_dir=session_path,
            model_path=model_path,
        )


def analyze_global_model_on_anchise(model_path, anchise_dir=ANCHISE_PATIENTS_DIR, output_dir=GLOBAL_ANCHISE_RESULTS_DIR):
    ensure_run_directories()
    os.makedirs(output_dir, exist_ok=True)

    for patient_id in tqdm(sorted(os.listdir(anchise_dir)), desc="Anchise"):
        patient_dir = os.path.join(anchise_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        coherent_txt_path = os.path.join(patient_dir, f"{patient_id}_adj.txt")
        incoherent_txt_path = os.path.join(patient_dir, f"{patient_id}_nadj.txt")
        if not os.path.exists(coherent_txt_path) or not os.path.exists(incoherent_txt_path):
            continue

        analyze_pairs_session(
            session_id=patient_id,
            coherent_txt_path=coherent_txt_path,
            incoherent_txt_path=incoherent_txt_path,
            output_dir=os.path.join(output_dir, patient_id),
            model_path=model_path,
        )


def calculate_all_values_for_loso_subject(subject_id, loso_root=NESTOR_LOSO_DIR):
    ensure_run_directories()

    split_dir = os.path.join(loso_root, subject_id)
    if not os.path.isdir(split_dir):
        print(f"Split dir mancante per soggetto {subject_id}: {split_dir}")
        return

    model_path = model_path_for_subject(subject_id)
    subject_test_dir = os.path.join(split_dir, "test")

    if not os.path.exists(model_path):
        print(f"Modello mancante per soggetto {subject_id}: {model_path}")
        return
    if not os.path.exists(subject_test_dir):
        print(f"Test dir mancante per soggetto {subject_id}: {subject_test_dir}")
        return

    print(f"VALUTAZIONE LOSO - SOGGETTO TEST: {subject_id}")
    analyze_coherence_on_subject_sessions(subject_test_dir, model_path)


def calculate_all_values_for_global_anchise():
    ensure_run_directories()

    model_path = model_path_for_subject(GLOBAL_MODEL_SUBJECT_ID)
    if not os.path.exists(model_path):
        print(f"Modello globale mancante: {model_path}")
        return

    print("VALUTAZIONE MODELLO GLOBALE SU ANCHISE")
    analyze_global_model_on_anchise(model_path=model_path)


def count_results_all(loso_root=NESTOR_LOSO_DIR, dementia_results_dir=GLOBAL_ANCHISE_RESULTS_DIR):
    ensure_run_directories()

    confusion_counts = {
        "control": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "dementia": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }
    gold_support = {"control": 0, "dementia": 0}

    if os.path.exists(loso_root):
        gold_support["control"] = collect_predictions_from_results(
            results_dir=loso_root,
            gold_label="control",
            confusion_counts=confusion_counts,
        )

    if os.path.exists(dementia_results_dir):
        gold_support["dementia"] = collect_predictions_from_flat_results(
            results_dir=dementia_results_dir,
            gold_label="dementia",
            confusion_counts=confusion_counts,
        )

    total_samples = gold_support["control"] + gold_support["dementia"]
    if total_samples == 0:
        print("Nessun risultato combinato trovato.")
        return None

    summary = {
        "run_id": RUN_ID,
        "run_root": RUN_ROOT,
        "total_samples": total_samples,
        "gold_support": gold_support,
        "missing_control_results": find_missing_control_results(loso_root=loso_root),
        "metrics": {},
    }

    total_correct = confusion_counts["control"]["tp"] + confusion_counts["dementia"]["tp"]
    summary["metrics"]["overall_accuracy"] = total_correct / total_samples

    if gold_support["control"] > 0:
        summary["metrics"]["control"] = compute_binary_metrics(**confusion_counts["control"])
    if gold_support["dementia"] > 0:
        summary["metrics"]["dementia"] = compute_binary_metrics(**confusion_counts["dementia"])
    if "control" in summary["metrics"] and "dementia" in summary["metrics"]:
        summary["metrics"]["harmonic_mean_accuracy_f1_control_f1_dementia"] = harmonic_mean(
            [
                summary["metrics"]["overall_accuracy"],
                summary["metrics"]["control"]["f1"],
                summary["metrics"]["dementia"]["f1"],
            ]
        )

    output_lines = [
        f"RUN_ID: {summary['run_id']}",
        f"RUN_ROOT: {summary['run_root']}",
        f"total_samples: {summary['total_samples']}",
        f"overall_accuracy: {summary['metrics']['overall_accuracy']:.6f}",
        f"gold_support: {summary['gold_support']}",
    ]
    if len(summary["missing_control_results"]) > 0:
        output_lines.append(f"missing_control_results: {summary['missing_control_results']}")

    if "control" in summary["metrics"]:
        control_metrics = summary["metrics"]["control"]
        output_lines.extend(
            [
                "control_metrics:",
                f"  precision: {control_metrics['precision']:.6f}",
                f"  recall: {control_metrics['recall']:.6f}",
                f"  f1: {control_metrics['f1']:.6f}",
                f"  accuracy: {control_metrics['accuracy']:.6f}",
                f"  tp: {control_metrics['tp']}",
                f"  fp: {control_metrics['fp']}",
                f"  tn: {control_metrics['tn']}",
                f"  fn: {control_metrics['fn']}",
            ]
        )
    else:
        output_lines.append("control_metrics: non calcolabili, nessun gold control disponibile")

    if "dementia" in summary["metrics"]:
        dementia_metrics = summary["metrics"]["dementia"]
        output_lines.extend(
            [
                "dementia_metrics:",
                f"  precision: {dementia_metrics['precision']:.6f}",
                f"  recall: {dementia_metrics['recall']:.6f}",
                f"  f1: {dementia_metrics['f1']:.6f}",
                f"  accuracy: {dementia_metrics['accuracy']:.6f}",
                f"  tp: {dementia_metrics['tp']}",
                f"  fp: {dementia_metrics['fp']}",
                f"  tn: {dementia_metrics['tn']}",
                f"  fn: {dementia_metrics['fn']}",
            ]
        )
    else:
        output_lines.append("dementia_metrics: non calcolabili, nessun gold dementia disponibile")

    if "harmonic_mean_accuracy_f1_control_f1_dementia" in summary["metrics"]:
        output_lines.append(
            "harmonic_mean_accuracy_f1_control_f1_dementia: "
            f"{summary['metrics']['harmonic_mean_accuracy_f1_control_f1_dementia']:.6f}"
        )
    else:
        output_lines.append("harmonic_mean_accuracy_f1_control_f1_dementia: non calcolabile, servono entrambe le classi")

    summary_output_path = os.path.join(RUN_ROOT, "final_metrics_summary.txt")
    with open(summary_output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
        f.write("\nraw_summary:\n")
        f.write(pformat(summary) + "\n")

    print("\n".join(output_lines))
    print(f"Summary salvato in: {summary_output_path}")
    return summary_output_path


def run_sequential_pipeline(target_subject_id=None):
    ensure_run_directories()
    state = load_run_state()

    if state["preprocessing"]["status"] != "done":
        state["preprocessing"]["status"] = "in_progress"
        save_run_state(state)
        preprocess_nestor_corpus()
        state["preprocessing"]["status"] = "done"
        save_run_state(state)

    if target_subject_id is None and state["global"]["status"] != "done":
        if state["global"]["status"] == "in_progress":
            cleanup_model_artifacts(GLOBAL_MODEL_SUBJECT_ID)
            cleanup_global_eval_results()

        state["global"]["status"] = "in_progress"
        save_run_state(state)

        train_global_model()
        calculate_all_values_for_global_anchise()
        cleanup_model_artifacts(GLOBAL_MODEL_SUBJECT_ID)

        state["global"]["status"] = "done"
        save_run_state(state)

    sessions_by_subject = list_sessions_by_subject(touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR)
    subject_ids = sorted(sessions_by_subject.keys())
    if target_subject_id is not None:
        if target_subject_id not in sessions_by_subject:
            raise ValueError(f"Subject ID non trovato: {target_subject_id}")
        subject_ids = [target_subject_id]

    current_subject = state["loso"].get("current_subject")
    completed_subjects = set(state["loso"].get("completed_subjects", []))

    ordered_subject_ids = []
    if current_subject in subject_ids and current_subject not in completed_subjects:
        ordered_subject_ids.append(current_subject)
    for subject_id in subject_ids:
        if subject_id == current_subject:
            continue
        if subject_id in completed_subjects:
            continue
        ordered_subject_ids.append(subject_id)

    print(f"RUN_ID: {RUN_ID}")
    print(f"RUN_ROOT: {RUN_ROOT}")
    print(f"Totale soggetti LOSO da processare: {len(ordered_subject_ids)}")

    for subject_id in ordered_subject_ids:
        state = load_run_state()
        state["loso"]["current_subject"] = subject_id
        save_run_state(state)

        cleanup_model_artifacts(subject_id)

        print(f"ALLENAMENTO LOSO - SOGGETTO TEST: {subject_id}")
        split_info = materialize_loso_split(
            subject_id=subject_id,
            sessions_by_subject=sessions_by_subject,
            touse_dir=PREPROCESSED_NESTOR_TOUSE_DIR,
            loso_root=NESTOR_LOSO_DIR,
        )
        train_dataset, dev_dataset = prepare_data_from_csv(split_info["train_csv_path"])
        train_model(train_dataset, dev_dataset, subject_id)
        calculate_all_values_for_loso_subject(subject_id, loso_root=NESTOR_LOSO_DIR)
        cleanup_model_artifacts(subject_id)

        state = load_run_state()
        completed = set(state["loso"].get("completed_subjects", []))
        completed.add(subject_id)
        state["loso"]["completed_subjects"] = sorted(completed)
        state["loso"]["current_subject"] = None
        save_run_state(state)

    summary_output_path = count_results_all()
    if summary_output_path is None:
        return
    state = load_run_state()
    state["final_metrics"]["status"] = "done"
    state["final_metrics"]["summary_output_path"] = summary_output_path
    save_run_state(state)


def parse_args():
    parser = argparse.ArgumentParser(description="Run coherence Nestor/Anchise review pipeline")
    parser.add_argument(
        "--step",
        choices=["train", "count", "preprocess", "pipeline"],
        default="train",
        help="Step della pipeline da eseguire",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Esegue il training LOSO solo per il subject ID indicato",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"RUN_ID: {RUN_ID}")
    print(f"RUN_ROOT: {RUN_ROOT}")

    if args.step == "preprocess":
        preprocess_nestor_corpus()
    elif args.step == "count":
        summary_output_path = count_results_all()
        if summary_output_path is not None:
            state = load_run_state()
            state["final_metrics"]["status"] = "done"
            state["final_metrics"]["summary_output_path"] = summary_output_path
            save_run_state(state)
    else:
        run_sequential_pipeline(target_subject_id=args.subject_id)
