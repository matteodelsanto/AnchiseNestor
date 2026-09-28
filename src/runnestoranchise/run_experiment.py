from __future__ import annotations

import csv
import gc
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Configura PyTorch per evitare frammentazione memoria.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add src path to PYTHONPATH for imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation_optimized import (  
    process_data_folders_multi_gpu,
    process_leave_one_out_multi_gpu,
)
from utils.models_training import leave_one_out_trainingV1, train_model  


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "anchise_nestor_balanced_completi"
DATASET_ROOT = PROJECT_ROOT / "resources" / "data" / "input" / DATASET_NAME
RUNTIME_ROOT = PROJECT_ROOT / "resources" / "data" / "input" / f"{DATASET_NAME}_runtime"

MODEL_TYPE = "gpt2"
MODEL_NAME = "LorenzoDeMattei/GePpeTto"
TEXT_FOLDER = "manual"
DISEASE_CLASS = "ad"
CONTROL_CLASS = "cn"

BATCH_SIZE = 16
MAX_EPOCHS = 1
SAVE_EVERY = 1
NUM_GPUS = 1
WINDOW = 20
LEAP = 0
GRADIENT_CHECKPOINTING = True
EXECUTION_ID = f"{datetime.now():%Y%m%d_%H%M%S_%f}_pid{os.getpid()}"


@dataclass(frozen=True)
class PatientDScore:
    true_label: str
    patient_id: str
    cn_ppl: float
    ad_ppl: float
    d_score: float

    @property
    def patient_key(self) -> str:
        return f"{self.true_label}/{self.patient_id}"


def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"anchise_nestor_experiment_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def cleanup_gpu_memory(logger: logging.Logger) -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for gpu_idx in range(torch.cuda.device_count()):
            with torch.cuda.device(gpu_idx):
                torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info(
            "GPU memory cleaned | allocated=%.2f GB reserved=%.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )


def patient_sort_key(path: Path) -> tuple[int, int | str]:
    if path.name.isdigit():
        return (0, int(path.name))
    return (1, path.name)


def patient_dirs(input_base_dir: Path) -> list[Path]:
    if not input_base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_base_dir}")
    return sorted(
        (path for path in input_base_dir.iterdir() if path.is_dir()),
        key=patient_sort_key,
    )


def model_name_for_class(class_name: str) -> str:
    return f"{class_name}_{TEXT_FOLDER}_{BATCH_SIZE}b_{MAX_EPOCHS}ep"


def final_model_dir(model_base_dir: Path) -> Path:
    return Path(f"{model_base_dir}_{MAX_EPOCHS}ep")


def train_dir_arg(input_dir: Path) -> str:
    return str(input_dir) + os.sep


def model_dir_is_complete(model_dir: Path) -> bool:
    if not model_dir.is_dir() or not (model_dir / "config.json").exists():
        return False
    has_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    has_tokenizer = (model_dir / "tokenizer.json").exists() or (
        (model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists()
    )
    return has_weights and has_tokenizer


def expected_global_ppl_file(output_base_dir: Path, patient_id: str, model_name: str) -> Path:
    return output_base_dir / patient_id / f"{patient_id}_modello_{model_name}_global_ppl_score.txt"


def expected_loo_ppl_file(output_base_dir: Path, patient_id: str, model_name: str) -> Path:
    return output_base_dir / patient_id / f"{patient_id}_modello_{model_name}_leave_one_out_ppl_score.txt"


def missing_global_ppl_patients(
    input_base_dir: Path,
    output_base_dir: Path,
    model_name: str,
) -> list[str]:
    return [
        patient_dir.name
        for patient_dir in patient_dirs(input_base_dir)
        if not expected_global_ppl_file(output_base_dir, patient_dir.name, model_name).exists()
    ]


def missing_loo_ppl_patients(
    input_base_dir: Path,
    output_base_dir: Path,
    model_name: str,
) -> list[str]:
    return [
        patient_dir.name
        for patient_dir in patient_dirs(input_base_dir)
        if not expected_loo_ppl_file(output_base_dir, patient_dir.name, model_name).exists()
    ]


def prepare_resume_input_dir(
    step_name: str,
    source_input_dir: Path,
    patient_ids: list[str],
) -> Path:
    resume_dir = RUNTIME_ROOT / "resume_inputs" / EXECUTION_ID / step_name
    if resume_dir.exists():
        shutil.rmtree(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)

    for patient_id in patient_ids:
        source_dir = source_input_dir / patient_id
        target_dir = resume_dir / patient_id
        target_dir.symlink_to(source_dir, target_is_directory=True)

    return resume_dir


def load_state(state_path: Path) -> dict[str, object]:
    if not state_path.exists():
        return {"steps": {}}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def mark_step(
    state_path: Path,
    state: dict[str, object],
    step_name: str,
    status: str,
    details: dict[str, object] | None = None,
) -> None:
    steps = state.setdefault("steps", {})
    assert isinstance(steps, dict)
    steps[step_name] = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
    }
    save_state(state_path, state)


def ensure_global_model(
    *,
    step_name: str,
    class_name: str,
    input_dir: Path,
    model_base_dir: Path,
    state_path: Path,
    state: dict[str, object],
    logger: logging.Logger,
) -> Path:
    target_dir = final_model_dir(model_base_dir)
    if model_dir_is_complete(target_dir):
        logger.info("Skipping %s; model already exists: %s", step_name, target_dir)
        mark_step(state_path, state, step_name, "completed", {"model_dir": str(target_dir), "resumed": True})
        return target_dir

    logger.info("%s", step_name)
    mark_step(state_path, state, step_name, "in_progress", {"model_dir": str(target_dir)})
    train_model(
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        train_set_file_path=train_dir_arg(input_dir),
        base_output_dir=str(model_base_dir),
        save_every=SAVE_EVERY,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )
    cleanup_gpu_memory(logger)

    if not model_dir_is_complete(target_dir):
        raise RuntimeError(f"Global {class_name} model was not created: {target_dir}")
    mark_step(state_path, state, step_name, "completed", {"model_dir": str(target_dir)})
    return target_dir


def ensure_loo_models(
    *,
    step_name: str,
    class_name: str,
    input_dir: Path,
    model_base_dir: Path,
    state_path: Path,
    state: dict[str, object],
    logger: logging.Logger,
) -> Path:
    target_base_dir = final_model_dir(model_base_dir)
    missing_patients = [
        patient_dir.name
        for patient_dir in patient_dirs(input_dir)
        if not model_dir_is_complete(target_base_dir / patient_dir.name)
    ]
    if not missing_patients:
        logger.info("Skipping %s; all LOO models already exist: %s", step_name, target_base_dir)
        mark_step(
            state_path,
            state,
            step_name,
            "completed",
            {"models_base_dir": str(target_base_dir), "resumed": True},
        )
        return target_base_dir

    logger.info("%s | missing_patients=%d", step_name, len(missing_patients))
    mark_step(
        state_path,
        state,
        step_name,
        "in_progress",
        {"models_base_dir": str(target_base_dir), "missing_patients": missing_patients},
    )

    if len(missing_patients) == len(patient_dirs(input_dir)):
        leave_one_out_trainingV1(
            MODEL_TYPE,
            MODEL_NAME,
            train_dir_arg(input_dir),
            str(model_base_dir),
            "test_text.txt",
            MAX_EPOCHS,
            BATCH_SIZE,
            save_every=SAVE_EVERY,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
        )
        cleanup_gpu_memory(logger)
    else:
        for patient_id in missing_patients:
            logger.info("Training %s LOO model on held-out patient: %s", class_name, patient_id)
            train_model(
                model_type=MODEL_TYPE,
                model_name=MODEL_NAME,
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                train_set_file_path=train_dir_arg(input_dir),
                base_output_dir=str(model_base_dir),
                loo_folder=patient_id,
                save_every=SAVE_EVERY,
                gradient_checkpointing=GRADIENT_CHECKPOINTING,
            )
            cleanup_gpu_memory(logger)

    still_missing = [
        patient_dir.name
        for patient_dir in patient_dirs(input_dir)
        if not model_dir_is_complete(target_base_dir / patient_dir.name)
    ]
    if still_missing:
        raise RuntimeError(
            f"Missing {len(still_missing)} LOO {class_name} models after training. "
            f"Examples: {still_missing[:10]}"
        )

    mark_step(state_path, state, step_name, "completed", {"models_base_dir": str(target_base_dir)})
    return target_base_dir


def ensure_global_ppl(
    *,
    step_name: str,
    label: str,
    model_dir: Path,
    input_dir: Path,
    output_dir: Path,
    state_path: Path,
    state: dict[str, object],
    logger: logging.Logger,
) -> None:
    model_name = model_dir.name
    missing_patients = missing_global_ppl_patients(input_dir, output_dir, model_name)
    if not missing_patients:
        logger.info("Skipping %s; all global PPL files already exist.", label)
        mark_step(state_path, state, step_name, "completed", {"output_dir": str(output_dir), "resumed": True})
        return

    logger.info("%s | missing_patients=%d", label, len(missing_patients))
    mark_step(
        state_path,
        state,
        step_name,
        "in_progress",
        {"model_dir": str(model_dir), "output_dir": str(output_dir), "missing_patients": missing_patients},
    )

    ppl_input_dir = (
        input_dir
        if len(missing_patients) == len(patient_dirs(input_dir))
        else prepare_resume_input_dir(step_name, input_dir, missing_patients)
    )
    process_data_folders_multi_gpu(
        model_type=MODEL_TYPE,
        model_dir=str(model_dir),
        input_base_dir=str(ppl_input_dir),
        output_base_dir=str(output_dir),
        test_file_name="test_text.txt",
        window=WINDOW,
        num_gpus=NUM_GPUS,
        leap=LEAP,
    )

    still_missing = missing_global_ppl_patients(input_dir, output_dir, model_name)
    if still_missing:
        raise RuntimeError(
            f"Missing {len(still_missing)} global PPL files after {label}. "
            f"Examples: {still_missing[:10]}"
        )
    mark_step(state_path, state, step_name, "completed", {"output_dir": str(output_dir)})


def ensure_loo_ppl(
    *,
    step_name: str,
    label: str,
    models_base_dir: Path,
    input_dir: Path,
    output_dir: Path,
    state_path: Path,
    state: dict[str, object],
    logger: logging.Logger,
) -> None:
    model_name = models_base_dir.name
    missing_patients = missing_loo_ppl_patients(input_dir, output_dir, model_name)
    if not missing_patients:
        logger.info("Skipping %s; all LOO PPL files already exist.", label)
        mark_step(state_path, state, step_name, "completed", {"output_dir": str(output_dir), "resumed": True})
        return

    logger.info("%s | missing_patients=%d", label, len(missing_patients))
    mark_step(
        state_path,
        state,
        step_name,
        "in_progress",
        {
            "models_base_dir": str(models_base_dir),
            "output_dir": str(output_dir),
            "missing_patients": missing_patients,
        },
    )

    ppl_input_dir = (
        input_dir
        if len(missing_patients) == len(patient_dirs(input_dir))
        else prepare_resume_input_dir(step_name, input_dir, missing_patients)
    )
    process_leave_one_out_multi_gpu(
        model_type=MODEL_TYPE,
        models_base_dir=str(models_base_dir),
        input_base_dir=str(ppl_input_dir),
        output_base_dir=str(output_dir),
        test_file_name="test_text.txt",
        window=WINDOW,
        num_gpus=NUM_GPUS,
        leap=LEAP,
    )

    still_missing = missing_loo_ppl_patients(input_dir, output_dir, model_name)
    if still_missing:
        raise RuntimeError(
            f"Missing {len(still_missing)} LOO PPL files after {label}. "
            f"Examples: {still_missing[:10]}"
        )
    mark_step(state_path, state, step_name, "completed", {"output_dir": str(output_dir)})


def read_ppl_score(score_path: Path) -> float:
    if not score_path.exists():
        raise FileNotFoundError(f"Perplexity file not found: {score_path}")

    raw_value = score_path.read_text(encoding="utf-8").strip()
    if not raw_value:
        raise ValueError(f"Perplexity file is empty: {score_path}")
    return float(raw_value)


def score_file_for_model(
    output_base_dir: Path,
    patient_id: str,
    model_name: str,
    *,
    leave_one_out: bool,
) -> Path:
    if leave_one_out:
        return expected_loo_ppl_file(output_base_dir, patient_id, model_name)
    return expected_global_ppl_file(output_base_dir, patient_id, model_name)


def load_class_d_scores(
    *,
    class_name: str,
    output_dir: Path,
    control_model_name: str,
    disease_model_name: str,
    control_score_is_loo: bool,
    disease_score_is_loo: bool,
) -> list[PatientDScore]:
    records: list[PatientDScore] = []
    missing_files: list[Path] = []

    for patient_dir in patient_dirs(output_dir):
        patient_id = patient_dir.name
        cn_score_path = score_file_for_model(
            output_dir,
            patient_id,
            control_model_name,
            leave_one_out=control_score_is_loo,
        )
        ad_score_path = score_file_for_model(
            output_dir,
            patient_id,
            disease_model_name,
            leave_one_out=disease_score_is_loo,
        )

        for score_path in (cn_score_path, ad_score_path):
            if not score_path.exists():
                missing_files.append(score_path)

        if cn_score_path.exists() and ad_score_path.exists():
            cn_ppl = read_ppl_score(cn_score_path)
            ad_ppl = read_ppl_score(ad_score_path)
            records.append(
                PatientDScore(
                    true_label=class_name,
                    patient_id=patient_id,
                    cn_ppl=cn_ppl,
                    ad_ppl=ad_ppl,
                    d_score=ad_ppl - cn_ppl,
                )
            )

    if missing_files:
        examples = "\n".join(str(path) for path in missing_files[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} perplexity files for {class_name}. "
            f"Examples:\n{examples}"
        )
    if not records:
        raise ValueError(f"No patient D scores loaded from {output_dir}")

    return records


def mean_d_score(records: list[PatientDScore]) -> float:
    if not records:
        raise ValueError("Cannot compute a mean D score from an empty training group")
    return sum(record.d_score for record in records) / len(records)


def classify_with_leave_one_out_means(
    *,
    control_records: list[PatientDScore],
    disease_records: list[PatientDScore],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_records = sorted(
        [*control_records, *disease_records],
        key=lambda record: (record.true_label, patient_sort_key(Path(record.patient_id))),
    )

    for patient in all_records:
        train_control = [
            record for record in control_records if record.patient_key != patient.patient_key
        ]
        train_disease = [
            record for record in disease_records if record.patient_key != patient.patient_key
        ]
        control_mean = mean_d_score(train_control)
        disease_mean = mean_d_score(train_disease)
        control_distance = abs(patient.d_score - control_mean)
        disease_distance = abs(patient.d_score - disease_mean)
        predicted_label = CONTROL_CLASS if control_distance <= disease_distance else DISEASE_CLASS

        rows.append(
            {
                "patient_key": patient.patient_key,
                "patient_id": patient.patient_id,
                "true_label": patient.true_label,
                "cn_ppl": patient.cn_ppl,
                "ad_ppl": patient.ad_ppl,
                "d_score": patient.d_score,
                "train_cn_mean_d": control_mean,
                "train_ad_mean_d": disease_mean,
                "distance_to_cn_mean": control_distance,
                "distance_to_ad_mean": disease_distance,
                "predicted_label": predicted_label,
                "correct": predicted_label == patient.true_label,
            }
        )

    return rows


def classification_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = [CONTROL_CLASS, DISEASE_CLASS]
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    by_label: dict[str, dict[str, float | int]] = {}
    confusion_matrix: dict[str, dict[str, int]] = {
        label: {predicted_label: 0 for predicted_label in labels} for label in labels
    }

    for label in labels:
        label_rows = [row for row in rows if row["true_label"] == label]
        label_correct = sum(1 for row in label_rows if row["correct"])
        by_label[label] = {
            "total": len(label_rows),
            "correct": label_correct,
            "accuracy": label_correct / len(label_rows) if label_rows else 0.0,
        }

    for row in rows:
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])
        confusion_matrix[true_label][predicted_label] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_label": by_label,
        "confusion_matrix": confusion_matrix,
    }


def d_score_reference_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    control_scores = [float(row["d_score"]) for row in rows if row["true_label"] == CONTROL_CLASS]
    disease_scores = [float(row["d_score"]) for row in rows if row["true_label"] == DISEASE_CLASS]
    control_mean = sum(control_scores) / len(control_scores)
    disease_mean = sum(disease_scores) / len(disease_scores)
    threshold = (control_mean + disease_mean) / 2.0

    return {
        "cn_mean_d": control_mean,
        "ad_mean_d": disease_mean,
        "threshold_d": threshold,
    }


def write_classification_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_key",
        "patient_id",
        "true_label",
        "cn_ppl",
        "ad_ppl",
        "d_score",
        "train_cn_mean_d",
        "train_ad_mean_d",
        "distance_to_cn_mean",
        "distance_to_ad_mean",
        "predicted_label",
        "correct",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_d_score_plot(
    rows: list[dict[str, object]],
    output_png: Path,
    output_pdf: Path | None = None,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No classification rows available for plotting")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    stats = d_score_reference_stats(rows)
    threshold = stats["threshold_d"]

    class_styles = {
        CONTROL_CLASS: {"color": "#2f6fbb", "label": "CN patients"},
        DISEASE_CLASS: {"color": "#c84c31", "label": "AD patients"},
    }
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            0 if row["true_label"] == DISEASE_CLASS else 1,
            patient_sort_key(Path(str(row["patient_id"]))),
        ),
    )
    x_positions = list(range(len(sorted_rows)))
    x_labels = [str(row["patient_key"]) for row in sorted_rows]

    fig_width = max(18.0, min(34.0, len(sorted_rows) * 0.12))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    for class_name in (CONTROL_CLASS, DISEASE_CLASS):
        class_points = [
            (index, row)
            for index, row in zip(x_positions, sorted_rows)
            if row["true_label"] == class_name
        ]
        xs = [index for index, _ in class_points]
        ys = [float(row["d_score"]) for _, row in class_points]
        colors = [
            class_styles[class_name]["color"] if row["correct"] else "#111111"
            for _, row in class_points
        ]
        markers = ["o" if row["correct"] else "x" for _, row in class_points]

        for marker in sorted(set(markers)):
            marker_xs = [x for x, row_marker in zip(xs, markers) if row_marker == marker]
            marker_ys = [y for y, row_marker in zip(ys, markers) if row_marker == marker]
            marker_colors = [
                color for color, row_marker in zip(colors, markers) if row_marker == marker
            ]
            if not marker_xs:
                continue
            label = class_styles[class_name]["label"] if marker == "o" else f"Wrong {class_name.upper()}"
            ax.scatter(
                marker_xs,
                marker_ys,
                c=marker_colors,
                marker=marker,
                s=44 if marker == "o" else 78,
                linewidths=1.6,
                alpha=0.78,
                label=label,
            )

    ax.axhline(
        stats["cn_mean_d"],
        color="#2f6fbb",
        linestyle="-",
        linewidth=2.2,
        label=f"CN mean D: {stats['cn_mean_d']:.3f}",
    )
    ax.axhline(
        stats["ad_mean_d"],
        color="#c84c31",
        linestyle="-",
        linewidth=2.2,
        label=f"AD mean D: {stats['ad_mean_d']:.3f}",
    )
    ax.axhline(
        threshold,
        color="#555555",
        linestyle="--",
        linewidth=2.0,
        label=f"Threshold: {threshold:.3f}",
    )

    ad_count = sum(1 for row in sorted_rows if row["true_label"] == DISEASE_CLASS)
    if 0 < ad_count < len(sorted_rows):
        ax.axvline(
            ad_count - 0.5,
            color="#999999",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )

    ax.set_xlim(-1, len(sorted_rows))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=5)
    ax.set_xlabel("Patient")
    ax.set_ylabel("D-score = AD perplexity - CN perplexity")
    ax.set_title("Anchise/Nestor leave-one-out D-score classification", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.grid(axis="x", linestyle="", alpha=0)

    label_x = len(sorted_rows) - 1
    ax.text(
        label_x,
        stats["cn_mean_d"],
        "CN mean",
        color="#2f6fbb",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        label_x,
        stats["ad_mean_d"],
        "AD mean",
        color="#c84c31",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        label_x,
        threshold,
        "threshold",
        color="#555555",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return stats


def run_delta_leave_one_out_classification(
    *,
    control_output_dir: Path,
    disease_output_dir: Path,
    control_model_name: str,
    disease_model_name: str,
    logger: logging.Logger,
) -> dict[str, object]:
    control_records = load_class_d_scores(
        class_name=CONTROL_CLASS,
        output_dir=control_output_dir,
        control_model_name=control_model_name,
        disease_model_name=disease_model_name,
        control_score_is_loo=True,
        disease_score_is_loo=False,
    )
    disease_records = load_class_d_scores(
        class_name=DISEASE_CLASS,
        output_dir=disease_output_dir,
        control_model_name=control_model_name,
        disease_model_name=disease_model_name,
        control_score_is_loo=False,
        disease_score_is_loo=True,
    )
    rows = classify_with_leave_one_out_means(
        control_records=control_records,
        disease_records=disease_records,
    )
    metrics = classification_metrics(rows)

    results_dir = (
        PROJECT_ROOT
        / "resources"
        / "data"
        / "results"
        / f"test_{MODEL_TYPE}_{DATASET_NAME}_all_patients_global_loo_nodev_{MAX_EPOCHS}ep"
    )
    output_csv = results_dir / "leave_one_out_predictions_from_ppl_table.csv"
    output_json = results_dir / "leave_one_out_metrics.json"
    output_plot_png = results_dir / "dscore_classification_plot.png"
    output_plot_pdf = results_dir / "dscore_classification_plot.pdf"

    write_classification_csv(rows, output_csv)
    plot_stats = create_d_score_plot(rows, output_plot_png, output_plot_pdf)
    metrics["d_score_reference"] = plot_stats
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "Classification completed | total=%d correct=%d accuracy=%.4f",
        metrics["total"],
        metrics["correct"],
        metrics["accuracy"],
    )
    logger.info("Classification CSV: %s", output_csv)
    logger.info("Classification metrics: %s", output_json)
    logger.info("Classification plot: %s", output_plot_png)

    return {
        **metrics,
        "results_dir": str(results_dir),
        "predictions_csv": str(output_csv),
        "metrics_json": str(output_json),
        "plot_png": str(output_plot_png),
        "plot_pdf": str(output_plot_pdf),
        "delta_definition": "D = AD perplexity - CN perplexity",
        "classification_rule": (
            "For each patient, remove that patient from its true-label train group, "
            "compare D with the remaining CN and AD mean D scores, and pick the closest mean."
        ),
    }


def build_classification_context(
    *,
    control_output_dir: Path,
    disease_output_dir: Path,
    control_model_name: str,
    disease_model_name: str,
) -> dict[str, object]:
    return {
        "control_dir": str(control_output_dir),
        "disease_dir": str(disease_output_dir),
        "control_model": control_model_name,
        "disease_model": disease_model_name,
        "input_layout": {
            "train_control": str(DATASET_ROOT / "train" / CONTROL_CLASS / TEXT_FOLDER),
            "train_disease": str(DATASET_ROOT / "train" / DISEASE_CLASS / TEXT_FOLDER),
        },
        "ppl_files": {
            "control_train_global_from_disease_model": f"*_modello_{disease_model_name}_global_ppl_score.txt",
            "control_train_leave_one_out_from_control_model": f"*_modello_{control_model_name}_leave_one_out_ppl_score.txt",
            "disease_train_global_from_control_model": f"*_modello_{control_model_name}_global_ppl_score.txt",
            "disease_train_leave_one_out_from_disease_model": f"*_modello_{disease_model_name}_leave_one_out_ppl_score.txt",
        },
        "delta_definition": "D = AD perplexity - CN perplexity",
    }


def main() -> int:
    logger = setup_logging()
    logger.info("Starting Anchise/Nestor train-only experiment")

    control_train_dir = DATASET_ROOT / "train" / CONTROL_CLASS / TEXT_FOLDER
    disease_train_dir = DATASET_ROOT / "train" / DISEASE_CLASS / TEXT_FOLDER

    models_root = PROJECT_ROOT / "resources" / "data" / "output" / "models" / DATASET_NAME / MODEL_TYPE
    ppl_root = PROJECT_ROOT / "resources" / "data" / f"output_{MODEL_TYPE}" / f"{DATASET_NAME}_w{WINDOW}_l{LEAP}"
    state_path = models_root / f"run_state_{TEXT_FOLDER}_{BATCH_SIZE}b_{MAX_EPOCHS}ep_w{WINDOW}_l{LEAP}.json"
    state = load_state(state_path)

    control_model_base = models_root / f"{CONTROL_CLASS}_{TEXT_FOLDER}_{BATCH_SIZE}b"
    disease_model_base = models_root / f"{DISEASE_CLASS}_{TEXT_FOLDER}_{BATCH_SIZE}b"
    control_loo_base = models_root / "leave_one_out" / f"{CONTROL_CLASS}_{TEXT_FOLDER}_{BATCH_SIZE}b"
    disease_loo_base = models_root / "leave_one_out" / f"{DISEASE_CLASS}_{TEXT_FOLDER}_{BATCH_SIZE}b"

    control_output_dir = ppl_root / "train" / CONTROL_CLASS / TEXT_FOLDER
    disease_output_dir = ppl_root / "train" / DISEASE_CLASS / TEXT_FOLDER

    try:
        logger.info("=" * 50)
        logger.info("PHASE 1: MODEL TRAINING")
        logger.info("=" * 50)
        control_global_dir = ensure_global_model(
            step_name="phase1_train_control_global",
            class_name=CONTROL_CLASS,
            input_dir=control_train_dir,
            model_base_dir=control_model_base,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        disease_global_dir = ensure_global_model(
            step_name="phase1_train_disease_global",
            class_name=DISEASE_CLASS,
            input_dir=disease_train_dir,
            model_base_dir=disease_model_base,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        logger.info("Training completed successfully!")

        logger.info("=" * 50)
        logger.info("PHASE 2: DEV PERPLEXITY PRODUCTION AND EPOCH SELECTION SKIPPED")
        logger.info("=" * 50)
        mark_step(
            state_path,
            state,
            "phase2_dev_epoch_selection",
            "skipped",
            {"reason": "Anchise/Nestor train-only layout has no dev split; using fixed epoch 10."},
        )

        logger.info("=" * 50)
        logger.info("PHASE 3: LOO MODELS + TRAIN PERPLEXITY PRODUCTION")
        logger.info("=" * 50)
        control_loo_dir = ensure_loo_models(
            step_name="phase3_train_control_loo_models",
            class_name=CONTROL_CLASS,
            input_dir=control_train_dir,
            model_base_dir=control_loo_base,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        disease_loo_dir = ensure_loo_models(
            step_name="phase3_train_disease_loo_models",
            class_name=DISEASE_CLASS,
            input_dir=disease_train_dir,
            model_base_dir=disease_loo_base,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        logger.info("LOO Training completed successfully!")

        ensure_global_ppl(
            step_name="phase3_ppl_control_on_disease_train",
            label="### CONTROL on DISEASE TRAIN #########",
            model_dir=control_global_dir,
            input_dir=disease_train_dir,
            output_dir=disease_output_dir,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        ensure_global_ppl(
            step_name="phase3_ppl_disease_on_control_train",
            label="### DISEASE on CONTROL TRAIN #########",
            model_dir=disease_global_dir,
            input_dir=control_train_dir,
            output_dir=control_output_dir,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        ensure_loo_ppl(
            step_name="phase3_ppl_leave_one_out_cn",
            label="######################## PPL LEAVE ONE OUT CN #############################",
            models_base_dir=control_loo_dir,
            input_dir=control_train_dir,
            output_dir=control_output_dir,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        ensure_loo_ppl(
            step_name="phase3_ppl_leave_one_out_disease",
            label="######################## PPL LEAVE ONE OUT DISEASE ########################",
            models_base_dir=disease_loo_dir,
            input_dir=disease_train_dir,
            output_dir=disease_output_dir,
            state_path=state_path,
            state=state,
            logger=logger,
        )
        logger.info("Perplexity production on train set using fixed-epoch models completed successfully!")

        classification_context = build_classification_context(
            control_output_dir=control_output_dir,
            disease_output_dir=disease_output_dir,
            control_model_name=control_loo_dir.name,
            disease_model_name=disease_loo_dir.name,
        )
        mark_step(
            state_path,
            state,
            "phase3_ready_for_classification",
            "completed",
            classification_context,
        )

        logger.info("=" * 50)
        logger.info("PHASE 4: CLASSIFICATION")
        logger.info("=" * 50)
        mark_step(
            state_path,
            state,
            "phase4_classification",
            "in_progress",
            classification_context,
        )
        classification_summary = run_delta_leave_one_out_classification(
            control_output_dir=control_output_dir,
            disease_output_dir=disease_output_dir,
            control_model_name=control_loo_dir.name,
            disease_model_name=disease_loo_dir.name,
            logger=logger,
        )
        mark_step(
            state_path,
            state,
            "phase4_classification",
            "completed",
            {**classification_context, **classification_summary},
        )
        return 0
    except Exception as exc:
        logger.error("Error during experiment: %s", exc)
        mark_step(state_path, state, "last_error", "failed", {"error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
