import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "resources" / "data" / "input" / "anchise_nestor_coherence"
OUTPUT_DIR = PROJECT_ROOT / "resources" / "data" / "output" / "coherence_na"
MODEL_BASE_PATH = PROJECT_ROOT / "resources" / "data" / "output" / "models" / "anchise_nestor_balanced_completi" / "gpt2"
CONTROL_GLOBAL_MODEL_DIR = MODEL_BASE_PATH / "cn_manual_16b_1ep"
CONTROL_LOO_MODEL_BASE_DIR = MODEL_BASE_PATH / "leave_one_out" / "cn_manual_16b_1ep"
RESULTS_CSV = OUTPUT_DIR / "coherence_classification_results.csv"
METRICS_JSON = OUTPUT_DIR / "coherence_classification_metrics.json"
FOLDER_LABELS = {
    "anchise": "dementia",
    "nestor": "control",
}
WINDOW_SIZE = 20
PPL_BATCH_SIZE = 64


def normalize_row(row):
    return {
        key.strip(): "" if value is None else value.strip()
        for key, value in row.items()
        if key is not None
    }


def patient_sort_key(path):
    if path.name.isdigit():
        return (0, int(path.name))
    return (1, path.name)


def get_pair_file_from_patient_dir(patient_dir, subject_id, pair_kind):
    suffix = "adj" if pair_kind == "adjacent" else "nadj"
    candidates = [
        patient_dir / (subject_id + "_" + suffix + ".txt"),
        patient_dir / (subject_id + ".txt_" + suffix + ".txt"),
        patient_dir / (subject_id + ".txt_" + suffix + ".tsv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    glob_patterns = [
        "*_" + suffix + ".txt",
        "*.txt_" + suffix + ".txt",
        "*.txt_" + suffix + ".tsv",
    ]
    matches = []
    for pattern in glob_patterns:
        matches.extend(sorted(patient_dir.glob(pattern)))
    return matches[0] if matches else None


def read_folder_transcriptions():
    transcriptions = []
    for source_folder, group in FOLDER_LABELS.items():
        source_dir = DATA_DIR / source_folder
        if not source_dir.is_dir():
            raise FileNotFoundError("Missing input folder: " + str(source_dir))

        patient_dirs = sorted(
            (path for path in source_dir.iterdir() if path.is_dir()),
            key=patient_sort_key,
        )
        for patient_dir in patient_dirs:
            subject_id = patient_dir.name
            transcriptions.append(
                {
                    "source_folder": source_folder,
                    "group": group,
                    "subject_id": subject_id,
                    "interview_id": subject_id,
                    "patient_dir": patient_dir,
                    "adjacent_file": get_pair_file_from_patient_dir(
                        patient_dir,
                        subject_id,
                        "adjacent",
                    ),
                    "non_adjacent_file": get_pair_file_from_patient_dir(
                        patient_dir,
                        subject_id,
                        "non_adjacent",
                    ),
                }
            )
    return transcriptions


def print_work_summary(transcriptions):
    groups = sorted(set(transcription["group"] for transcription in transcriptions))
    print("Work summary:", flush=True)
    print("  total transcriptions:", len(transcriptions), flush=True)
    for group in groups:
        group_transcriptions = [
            transcription
            for transcription in transcriptions
            if transcription["group"] == group
        ]
        subjects = set(transcription_prefix(transcription) for transcription in group_transcriptions)
        print(
            "  "
            + group
            + ": "
            + str(len(group_transcriptions))
            + " transcriptions, "
            + str(len(subjects))
            + " subjects",
            flush=True,
        )


def transcription_prefix(transcription):
    if transcription.get("source_folder"):
        return transcription["source_folder"] + "/" + transcription["subject_id"]
    return transcription["subject_id"] + "-" + transcription["interview_id"]


def get_pair_file(transcription, pair_kind):
    key = "adjacent_file" if pair_kind == "adjacent" else "non_adjacent_file"
    if transcription.get(key) is not None:
        return transcription[key]

    prefix = transcription_prefix(transcription)
    suffix = "adj" if pair_kind == "adjacent" else "nadj"
    txt_path = DATA_DIR / (prefix + ".txt_" + suffix + ".txt")
    tsv_path = DATA_DIR / (prefix + ".txt_" + suffix + ".tsv")

    if tsv_path.exists():
        return tsv_path
    if txt_path.exists():
        return txt_path
    return None


def iter_pair_texts_from_tsv(path):
    with path.open(newline="") as file_handle:
        for row in csv.DictReader(file_handle, delimiter="\t"):
            utterance_1 = row["U1"].strip()
            utterance_2 = row["U2"].strip()
            if utterance_1 and utterance_2:
                yield utterance_1 + " " + utterance_2


def iter_pair_texts_from_txt(path):
    with path.open(newline="") as file_handle:
        for row in csv.reader(file_handle, delimiter="\t"):
            if len(row) < 2:
                continue
            utterance_1 = row[0].strip()
            utterance_2 = row[1].strip()
            if utterance_1 and utterance_2:
                yield utterance_1 + " " + utterance_2


def read_pair_texts(path):
    if path is None:
        return []
    if path.suffix == ".tsv":
        return list(iter_pair_texts_from_tsv(path))
    return list(iter_pair_texts_from_txt(path))


def model_dir_for_transcription(transcription):
    source_folder = transcription.get("source_folder")
    if source_folder == "nestor":
        return CONTROL_LOO_MODEL_BASE_DIR / transcription["subject_id"]
    if source_folder == "anchise":
        return CONTROL_GLOBAL_MODEL_DIR

    if transcription["group"] == "control":
        return CONTROL_LOO_MODEL_BASE_DIR / transcription["subject_id"]
    return CONTROL_GLOBAL_MODEL_DIR


def load_model(model_dir):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if not model_dir.is_dir():
        raise FileNotFoundError("Model directory not found: " + str(model_dir))

    print("Loading model from:", model_dir, flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def compute_text_avg_perplexity_window_fast(subject_interview, tokenizer, model, window_size=None):
    encodings = tokenizer(subject_interview, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    sequence_length = input_ids.size(0)
    max_length = 50 if window_size is None else window_size

    windows = []
    lengths = []
    for i in range(1, sequence_length):
        begin_loc = max(i + 1 - max_length, 0)
        end_loc = min(i + 1, sequence_length)
        window = input_ids[begin_loc:end_loc]
        windows.append(window)
        lengths.append(window.size(0))

    if not windows:
        raise ValueError("Cannot compute perplexity for a text shorter than two tokens.")

    losses = []
    for start in range(0, len(windows), PPL_BATCH_SIZE):
        batch_windows = windows[start:start + PPL_BATCH_SIZE]
        batch_lengths = lengths[start:start + PPL_BATCH_SIZE]
        batch_max_length = max(batch_lengths)

        batch_input_ids = torch.full(
            (len(batch_windows), batch_max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(batch_windows), batch_max_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(batch_windows), batch_max_length),
            -100,
            dtype=torch.long,
        )

        for row, (window, length) in enumerate(zip(batch_windows, batch_lengths)):
            batch_input_ids[row, :length] = window
            attention_mask[row, :length] = 1
            labels[row, length - 1] = window[-1]

        with torch.no_grad():
            logits = model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
            ).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())
        losses.extend(token_losses.sum(dim=1).tolist())

    return torch.exp(torch.tensor(losses).sum() / sequence_length)


def compute_coherences(texts, tokenizer, model, pair_kind):
    coherences = []
    for index, text in enumerate(texts, start=1):
        ppl = compute_text_avg_perplexity_window_fast(
            text,
            tokenizer,
            model,
            window_size=WINDOW_SIZE,
        )
        coherences.append(1 - float(ppl))
        if index % 25 == 0 or index == len(texts):
            print("    " + pair_kind + " progress:", index, "/", len(texts), flush=True)
    return coherences


def compute_transcription_result(transcription, tokenizer, model, model_dir):
    adjacent_file = get_pair_file(transcription, "adjacent")
    non_adjacent_file = get_pair_file(transcription, "non_adjacent")

    adjacent_texts = read_pair_texts(adjacent_file)
    non_adjacent_texts = read_pair_texts(non_adjacent_file)

    if adjacent_file is not None:
        print("  Adjacent file:", display_path(adjacent_file), "pairs:", len(adjacent_texts), flush=True)
    else:
        print("  Adjacent file: missing", flush=True)

    adjacent_coherences = compute_coherences(adjacent_texts, tokenizer, model, "adjacent")

    if non_adjacent_file is not None:
        print("  Non-adjacent file:", display_path(non_adjacent_file), "pairs:", len(non_adjacent_texts), flush=True)
    else:
        print("  Non-adjacent file: missing", flush=True)

    non_adjacent_coherences = compute_coherences(non_adjacent_texts, tokenizer, model, "non-adjacent")

    avg_adjacent = mean(adjacent_coherences) if adjacent_coherences else None
    avg_non_adjacent = mean(non_adjacent_coherences) if non_adjacent_coherences else None

    if avg_adjacent is None or avg_non_adjacent is None:
        predicted_label = "not_classified"
        predicted_group = "not_classified"
    elif avg_adjacent > avg_non_adjacent:
        predicted_label = "healthy"
        predicted_group = "control"
    else:
        predicted_label = "impaired"
        predicted_group = "dementia"

    return {
        "source_folder": transcription.get("source_folder", ""),
        "group": transcription["group"],
        "subject_id": transcription["subject_id"],
        "interview_id": transcription["interview_id"],
        "model_dir": str(model_dir),
        "avg_adjacent": avg_adjacent,
        "avg_non_adjacent": avg_non_adjacent,
        "predicted_label": predicted_label,
        "predicted_group": predicted_group,
        "n_adjacent": len(adjacent_coherences),
        "n_non_adjacent": len(non_adjacent_coherences),
        "adjacent_file": "" if adjacent_file is None else display_path(adjacent_file),
        "non_adjacent_file": "" if non_adjacent_file is None else display_path(non_adjacent_file),
    }


def display_path(path):
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def csv_value(value):
    return "" if value is None else value


def result_key(result):
    return (
        result.get("source_folder", ""),
        result["group"],
        result["subject_id"],
        result["interview_id"],
    )


def transcription_key(transcription):
    return (
        transcription.get("source_folder", ""),
        transcription["group"],
        transcription["subject_id"],
        transcription["interview_id"],
    )


def parse_csv_result(row):
    row = normalize_row(row)
    return {
        "source_folder": row.get("source_folder", ""),
        "group": row["group"],
        "subject_id": row["subject_id"],
        "interview_id": row["interview_id"],
        "model_dir": row["model_dir"],
        "avg_adjacent": None if row["avg_adjacent"] == "" else float(row["avg_adjacent"]),
        "avg_non_adjacent": None if row["avg_non_adjacent"] == "" else float(row["avg_non_adjacent"]),
        "predicted_label": row["predicted_label"],
        "predicted_group": row["predicted_group"],
        "n_adjacent": int(row["n_adjacent"]) if row["n_adjacent"] else 0,
        "n_non_adjacent": int(row["n_non_adjacent"]) if row["n_non_adjacent"] else 0,
        "adjacent_file": row["adjacent_file"],
        "non_adjacent_file": row["non_adjacent_file"],
    }


def read_existing_results():
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return []

    with RESULTS_CSV.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return []
        return [parse_csv_result(row) for row in reader]


def compute_one_vs_rest_metrics(results, positive_class):
    tp = fp = tn = fn = 0
    for result in results:
        gold_positive = result["group"] == positive_class
        pred_positive = result["predicted_group"] == positive_class

        if gold_positive and pred_positive:
            tp += 1
        elif not gold_positive and pred_positive:
            fp += 1
        elif not gold_positive and not pred_positive:
            tn += 1
        elif gold_positive and not pred_positive:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else 0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def harmonic_mean(values):
    if not values or any(value == 0 for value in values):
        return 0
    return len(values) / sum(1 / value for value in values)


def build_metrics(results):
    classified_results = [
        result
        for result in results
        if result["predicted_group"] in {"control", "dementia"}
    ]

    gold_support = {
        "control": sum(1 for result in classified_results if result["group"] == "control"),
        "dementia": sum(1 for result in classified_results if result["group"] == "dementia"),
    }
    control_metrics = compute_one_vs_rest_metrics(classified_results, "control")
    dementia_metrics = compute_one_vs_rest_metrics(classified_results, "dementia")

    hm = harmonic_mean(
        [
            control_metrics["accuracy"],
            control_metrics["f1"],
            dementia_metrics["f1"],
        ]
    )

    return {
        "gold_support": gold_support,
        "control_metrics": control_metrics,
        "dementia_metrics": dementia_metrics,
        "harmonic_mean_accuracy_f1_control_f1_dementia": round(hm, 6),
    }


def write_metrics(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(results)
    with METRICS_JSON.open("w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print("Wrote metrics JSON to:", METRICS_JSON, flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transcriptions = read_folder_transcriptions()
    print_work_summary(transcriptions)
    existing_results = read_existing_results()
    completed = set(result_key(result) for result in existing_results)
    results = list(existing_results)
    current_model_dir = None
    tokenizer = None
    model = None

    fieldnames = [
        "source_folder",
        "group",
        "subject_id",
        "interview_id",
        "model_dir",
        "avg_adjacent",
        "avg_non_adjacent",
        "predicted_label",
        "predicted_group",
        "n_adjacent",
        "n_non_adjacent",
        "adjacent_file",
        "non_adjacent_file",
    ]

    remaining = [
        transcription
        for transcription in transcriptions
        if transcription_key(transcription) not in completed
    ]
    print("Resume summary:", flush=True)
    print("  existing completed results:", len(existing_results), flush=True)
    print("  remaining transcriptions:", len(remaining), flush=True)

    open_mode = "a" if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0 else "w"
    with RESULTS_CSV.open(open_mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if open_mode == "w":
            writer.writeheader()

        for index, transcription in enumerate(transcriptions, start=1):
            if transcription_key(transcription) in completed:
                print(
                    "Skipping completed transcription "
                    + str(index)
                    + "/"
                    + str(len(transcriptions))
                    + ": "
                    + transcription_prefix(transcription),
                    flush=True,
                )
                continue

            prefix = transcription_prefix(transcription)
            model_dir = model_dir_for_transcription(transcription)
            print(
                "Transcription "
                + str(index)
                + "/"
                + str(len(transcriptions))
                + ": "
                + prefix
                + " group="
                + transcription["group"],
                flush=True,
            )

            if model_dir != current_model_dir:
                tokenizer, model = load_model(model_dir)
                current_model_dir = model_dir

            result = compute_transcription_result(transcription, tokenizer, model, model_dir)
            writer.writerow({key: csv_value(result[key]) for key in fieldnames})
            csv_file.flush()
            results.append(result)

            print(
                "RESULT,"
                + result["source_folder"]
                + ","
                + result["group"]
                + ","
                + result["subject_id"]
                + ","
                + result["interview_id"]
                + ","
                + result["predicted_label"],
                flush=True,
            )

    print("Wrote results CSV to:", RESULTS_CSV, flush=True)
    write_metrics(results)


if __name__ == "__main__":
    main()
