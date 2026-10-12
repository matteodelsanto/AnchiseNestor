"""Esperimento Anchise/Nestor sui 5 fold (resources/data/input/anchise_nestor_fold_k).

Produce per ogni fold gli stessi file di run_experiment.py lanciato con DATASET_NAME = "anchise_nestor_fold_k"
(PPL in resources/data/output_gpt2/anchise_nestor_fold_k_w20_l0, risultati in resources/data/results/...),
ma allenando ogni modello una sola volta:

- cn globale (uno solo, Nestor e' identico in tutti i fold) -> PPL globale degli AD di tutti i fold
- cn LOO (28) -> PPL LOO di ogni CN, scritta in tutti i fold
- ad globale di ogni fold -> PPL globale dei CN del fold
- ad LOO di ogni fold (28 per fold) -> PPL LOO dell'AD escluso

Ogni modello viene cancellato appena le sue PPL sono scritte. I training girano in processi separati, in
parallelo finche' c'e' memoria GPU libera (misurata durante l'esecuzione, contando anche gli altri processi
sulla GPU); un job che va in OOM viene rimesso in coda. Il training usa utils.models_training.train_model
con gli stessi argomenti di run_experiment.py; la PPL replica utils.perplexity.compute_global_perplexity
con le finestre impilate in batch (PPL_BATCHED = False usa la funzione originale).

train_model legge i pazienti nell'ordine di os.walk, che dipende dal filesystem (su ext4 da un hash con un
seme diverso per ogni disco): copiata su un'altra macchina, la stessa cartella darebbe altre batch. L'ordine
letto sulla macchina di riferimento e' salvato in fold_walk_order.json e imposto a train_model ovunque.

Uso (da src/): python -m runnestoranchise.run_experiment_fold [--dry-run]
               python -m runnestoranchise.run_experiment_fold --record-walk-order  (solo sulla macchina di riferimento)
"""

from __future__ import annotations

import csv
import fcntl
import getpass
import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import socket
import statistics
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.connection import wait as wait_connections
from pathlib import Path

# Configura PyTorch per evitare frammentazione memoria.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add src path to PYTHONPATH for imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runnestoranchise import run_experiment as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLD_DATASET_PREFIX = "anchise_nestor_fold_"
FOLDS = tuple(range(1, 6))
FOLDS_RUN_NAME = "anchise_nestor_folds"
WALK_ORDER_PATH = Path(__file__).with_name("fold_walk_order.json")

MODEL_TYPE = "gpt2"
MODEL_NAME = "LorenzoDeMattei/GePpeTto"
TEXT_FOLDER = "manual"
DISEASE_CLASS = "ad"
CONTROL_CLASS = "cn"
TEST_FILE_NAME = "test_text.txt"

BATCH_SIZE = 16
MAX_EPOCHS = 30
SAVE_EVERY = MAX_EPOCHS  # si salva solo il modello finale, cancellato dopo le sue PPL
WINDOW = 20
LEAP = 0
GRADIENT_CHECKPOINTING = True

# Esecuzione: non cambiano i modelli allenati.
NUM_GPUS = None  # None = tutte le GPU visibili (limitabili con CUDA_VISIBLE_DEVICES); ogni modello usa una GPU
MAX_PARALLEL_JOBS = 4
GPU_MEMORY_MARGIN_GB = 2.0  # memoria GPU lasciata libera oltre al picco stimato dei job
WORKER_CONTEXT_GB = 0.6  # contesto CUDA di un nuovo processo worker
FIRST_JOB_MIN_FREE_GB = 8.0  # finche' non c'e' un picco misurato i job partono uno alla volta
DISK_PER_JOB_GB = 1.5
MIN_FREE_DISK_GB = 5.0
PPL_BATCHED = True
PPL_WINDOWS_PER_BATCH = 64  # logits fp32: 64 x 20 x 30000 x 4 B = ~150 MB
OOM_COOLDOWN_SECONDS = 120
MAX_OOM_RETRIES = 20
MAX_ERROR_RETRIES = 2
SUCCESSES_TO_RAISE_CAP = 3
POLL_SECONDS = 5
STATUS_LOG_SECONDS = 300
# Prenotazioni di memoria condivise tra i run lanciati in contemporanea sulla stessa macchina (anche con
# epoche diverse): un job appena partito non ha ancora allocato, e senza prenotazione un altro run lo
# ignorerebbe e lancerebbe un job sulla stessa GPU.
GPU_LEDGER_PATH = Path(tempfile.gettempdir()) / f"anchise_nestor_gpu_ledger_{getpass.getuser()}.json"
GPU_LEDGER_STALE_SECONDS = 30

GB = 1024**3


@dataclass(frozen=True)
class Settings:
    model_type: str
    model_name: str
    max_epochs: int
    batch_size: int
    save_every: int
    gradient_checkpointing: bool
    window: int
    leap: int
    ppl_batched: bool
    ppl_windows_per_batch: int


@dataclass(frozen=True)
class PplTarget:
    text_path: str
    output_paths: tuple[str, ...]

    def is_complete(self) -> bool:
        return all(Path(path).exists() for path in self.output_paths)


@dataclass(frozen=True)
class Job:
    key: str
    train_dir: str
    train_order: tuple[str, ...]
    model_base_dir: str
    loo_folder: str | None
    targets: tuple[PplTarget, ...]
    settings: Settings

    @property
    def final_dir(self) -> Path:
        return Path(f"{self.model_base_dir}_{self.settings.max_epochs}ep")

    @property
    def model_dir(self) -> Path:
        return self.final_dir / self.loo_folder if self.loo_folder else self.final_dir

    def missing_targets(self) -> tuple[PplTarget, ...]:
        return tuple(target for target in self.targets if not target.is_complete())


# --------------------------------------------------------------------------- #
#                                   Paths                                     #
# --------------------------------------------------------------------------- #

def fold_name(fold: int) -> str:
    return f"{FOLD_DATASET_PREFIX}{fold}"


def fold_class_dir(fold: int, class_name: str) -> Path:
    return PROJECT_ROOT / "resources" / "data" / "input" / fold_name(fold) / "train" / class_name / TEXT_FOLDER


def fold_ppl_dir(fold: int, class_name: str) -> Path:
    ppl_root = PROJECT_ROOT / "resources" / "data" / f"output_{MODEL_TYPE}" / f"{fold_name(fold)}_w{WINDOW}_l{LEAP}"
    return ppl_root / "train" / class_name / TEXT_FOLDER


def results_dir_for(dataset_name: str) -> Path:
    return (
        PROJECT_ROOT
        / "resources"
        / "data"
        / "results"
        / f"test_{MODEL_TYPE}_{dataset_name}_all_patients_global_loo_nodev_{MAX_EPOCHS}ep"
    )


def models_root() -> Path:
    return PROJECT_ROOT / "resources" / "data" / "output" / "models" / FOLDS_RUN_NAME / MODEL_TYPE


def model_base_name(class_name: str) -> str:
    return f"{class_name}_{TEXT_FOLDER}_{BATCH_SIZE}b"


def ppl_model_name(class_name: str) -> str:
    # Nome che run_experiment mette nei file PPL: basename della cartella del modello finale.
    return f"{model_base_name(class_name)}_{MAX_EPOCHS}ep"


def run_tag() -> str:
    return f"{TEXT_FOLDER}_{BATCH_SIZE}b_{MAX_EPOCHS}ep_w{WINDOW}_l{LEAP}"


def patient_ids(directory: Path) -> list[str]:
    return [path.name for path in base.patient_dirs(directory)]


def relative_key(directory: Path) -> str:
    return directory.relative_to(PROJECT_ROOT).as_posix()


def filesystem_walk_order(directory: Path) -> list[str]:
    # Ordine in cui os.walk (e quindi train_model) visita le cartelle paziente su questo filesystem.
    return [entry.name for entry in os.scandir(directory) if entry.is_dir()]


def record_walk_order() -> int:
    if WALK_ORDER_PATH.exists():
        print(f"{WALK_ORDER_PATH} already exists; delete it first to record a new order")
        return 1
    content = {
        "recorded_on": socket.gethostname(),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "note": "os.walk order of each training directory on the reference machine; "
        "run_experiment_fold imposes it on train_model on every machine.",
        "dirs": {
            relative_key(fold_class_dir(fold, class_name)): filesystem_walk_order(fold_class_dir(fold, class_name))
            for fold in FOLDS
            for class_name in (CONTROL_CLASS, DISEASE_CLASS)
        },
    }
    WALK_ORDER_PATH.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Recorded the order of {len(content['dirs'])} directories in {WALK_ORDER_PATH}")
    return 0


def load_walk_order() -> dict[str, list[str]]:
    if not WALK_ORDER_PATH.exists():
        raise FileNotFoundError(
            f"{WALK_ORDER_PATH} not found: run --record-walk-order on the reference machine and copy the file here"
        )
    return json.loads(WALK_ORDER_PATH.read_text(encoding="utf-8"))["dirs"]


# --------------------------------------------------------------------------- #
#                                   Setup                                     #
# --------------------------------------------------------------------------- #

def setup_logging(timestamp: str) -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"anchise_nestor_folds_{MAX_EPOCHS}ep_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def current_settings() -> Settings:
    return Settings(
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_every=SAVE_EVERY,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        window=WINDOW,
        leap=LEAP,
        ppl_batched=PPL_BATCHED,
        ppl_windows_per_batch=PPL_WINDOWS_PER_BATCH,
    )


def validate_folds(walk_order: dict[str, list[str]]) -> None:
    if MODEL_TYPE != "gpt2":
        raise ValueError("run_experiment_fold supports only MODEL_TYPE = 'gpt2'")
    if SAVE_EVERY != MAX_EPOCHS:
        raise ValueError("SAVE_EVERY must equal MAX_EPOCHS: intermediate checkpoints would never be deleted")

    for fold in FOLDS:
        for class_name in (CONTROL_CLASS, DISEASE_CLASS):
            directory = fold_class_dir(fold, class_name)
            if not directory.is_dir():
                raise FileNotFoundError(f"Fold input directory not found: {directory}")
            key = relative_key(directory)
            if key not in walk_order:
                raise ValueError(f"{key} is missing from {WALK_ORDER_PATH.name}")
            if sorted(walk_order[key]) != sorted(patient_ids(directory)):
                raise ValueError(f"Patients in {directory} differ from those recorded in {WALK_ORDER_PATH.name}")

    # I modelli cn sono condivisi: i fold devono avere gli stessi testi cn nello stesso ordine di lettura,
    # cosi' il modello e' quello che run_experiment allenerebbe su ogni fold.
    reference_dir = fold_class_dir(FOLDS[0], CONTROL_CLASS)
    reference_order = walk_order[relative_key(reference_dir)]
    reference_texts = {pid: (reference_dir / pid / TEST_FILE_NAME).read_bytes() for pid in reference_order}
    for fold in FOLDS[1:]:
        directory = fold_class_dir(fold, CONTROL_CLASS)
        if walk_order[relative_key(directory)] != reference_order:
            raise ValueError(f"{directory} has a different recorded cn order than {reference_dir}")
        for pid in reference_order:
            if (directory / pid / TEST_FILE_NAME).read_bytes() != reference_texts[pid]:
                raise ValueError(f"cn patient {pid} differs between {reference_dir} and {directory}")


def build_jobs(settings: Settings, walk_order: dict[str, list[str]]) -> list[Job]:
    root = models_root()
    control_dir = fold_class_dir(FOLDS[0], CONTROL_CLASS)
    control_order = tuple(walk_order[relative_key(control_dir)])
    control_ids = patient_ids(control_dir)
    control_model = ppl_model_name(CONTROL_CLASS)
    disease_model = ppl_model_name(DISEASE_CLASS)

    jobs = [
        Job(
            key="cn_global",
            train_dir=str(control_dir),
            train_order=control_order,
            model_base_dir=str(root / model_base_name(CONTROL_CLASS)),
            loo_folder=None,
            targets=tuple(
                PplTarget(
                    text_path=str(fold_class_dir(fold, DISEASE_CLASS) / pid / TEST_FILE_NAME),
                    output_paths=(
                        str(base.expected_global_ppl_file(fold_ppl_dir(fold, DISEASE_CLASS), pid, control_model)),
                    ),
                )
                for fold in FOLDS
                for pid in patient_ids(fold_class_dir(fold, DISEASE_CLASS))
            ),
            settings=settings,
        )
    ]
    for pid in control_ids:
        jobs.append(
            Job(
                key=f"cn_loo/{pid}",
                train_dir=str(control_dir),
                train_order=control_order,
                model_base_dir=str(root / "leave_one_out" / model_base_name(CONTROL_CLASS)),
                loo_folder=pid,
                targets=(
                    PplTarget(
                        text_path=str(control_dir / pid / TEST_FILE_NAME),
                        output_paths=tuple(
                            str(base.expected_loo_ppl_file(fold_ppl_dir(fold, CONTROL_CLASS), pid, control_model))
                            for fold in FOLDS
                        ),
                    ),
                ),
                settings=settings,
            )
        )

    for fold in FOLDS:
        disease_dir = fold_class_dir(fold, DISEASE_CLASS)
        disease_order = tuple(walk_order[relative_key(disease_dir)])
        fold_control_dir = fold_class_dir(fold, CONTROL_CLASS)
        fold_root = root / fold_name(fold)
        jobs.append(
            Job(
                key=f"fold_{fold}/ad_global",
                train_dir=str(disease_dir),
                train_order=disease_order,
                model_base_dir=str(fold_root / model_base_name(DISEASE_CLASS)),
                loo_folder=None,
                targets=tuple(
                    PplTarget(
                        text_path=str(fold_control_dir / pid / TEST_FILE_NAME),
                        output_paths=(
                            str(base.expected_global_ppl_file(fold_ppl_dir(fold, CONTROL_CLASS), pid, disease_model)),
                        ),
                    )
                    for pid in patient_ids(fold_control_dir)
                ),
                settings=settings,
            )
        )
        for pid in patient_ids(disease_dir):
            jobs.append(
                Job(
                    key=f"fold_{fold}/ad_loo/{pid}",
                    train_dir=str(disease_dir),
                    train_order=disease_order,
                    model_base_dir=str(fold_root / "leave_one_out" / model_base_name(DISEASE_CLASS)),
                    loo_folder=pid,
                    targets=(
                        PplTarget(
                            text_path=str(disease_dir / pid / TEST_FILE_NAME),
                            output_paths=(
                                str(base.expected_loo_ppl_file(fold_ppl_dir(fold, DISEASE_CLASS), pid, disease_model)),
                            ),
                        ),
                    ),
                    settings=settings,
                )
            )
    return jobs


def fold_job_keys(jobs: list[Job], fold: int) -> set[str]:
    return {job.key for job in jobs if job.key.startswith(("cn_", f"fold_{fold}/"))}


# --------------------------------------------------------------------------- #
#                                 Perplexity                                  #
# --------------------------------------------------------------------------- #

def batched_global_perplexities(texts, model, tokenizer, window_size, device, leap, windows_per_batch):
    """Stessa aritmetica di utils.perplexity.compute_global_perplexity (finestre con stride 1, la prima
    pesa trg_len = lunghezza finestra, le altre solo l'ultimo token), con le finestre impilate in batch."""
    import torch
    import torch.nn.functional as F

    max_length = window_size if window_size > 0 else model.config.n_positions
    encoded = [tokenizer(text, return_tensors="pt").to(device).input_ids[0] for text in texts]

    windows_by_length: dict[int, list[tuple[int, int]]] = {}
    offsets, counts = [], []
    total_windows = 0
    for text_idx, ids in enumerate(encoded):
        seq_len = ids.size(0)
        length = min(max_length, seq_len)
        count = 1 if seq_len <= max_length else 1 + seq_len - max_length
        offsets.append(total_windows)
        counts.append(count)
        total_windows += count
        windows_by_length.setdefault(length, []).extend((text_idx, begin) for begin in range(count))

    nlls = torch.empty(total_windows, dtype=torch.float32, device=device)
    with torch.no_grad():
        for length, windows in windows_by_length.items():
            for start in range(0, len(windows), windows_per_batch):
                chunk = windows[start : start + windows_per_batch]
                input_ids = torch.stack([encoded[text_idx][begin : begin + length] for text_idx, begin in chunk])
                logits = model(input_ids).logits.float()

                tail_rows = [row for row, (_, begin) in enumerate(chunk) if begin > 0]
                if tail_rows:
                    rows = torch.tensor(tail_rows, device=device)
                    positions = torch.tensor([offsets[chunk[row][0]] + chunk[row][1] for row in tail_rows], device=device)
                    nlls[positions] = F.cross_entropy(
                        logits[rows, length - 2], input_ids[rows, length - 1], reduction="none"
                    )
                for row, (text_idx, begin) in enumerate(chunk):
                    if begin == 0:
                        labels = F.pad(input_ids[row : row + 1], (0, 1), value=-100)[..., 1:]
                        loss = F.cross_entropy(
                            logits[row : row + 1].reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
                        )
                        nlls[offsets[text_idx]] = loss * length

        ppls = []
        for text_idx, ids in enumerate(encoded):
            text_nlls = nlls[offsets[text_idx] : offsets[text_idx] + counts[text_idx]][leap:].clone()
            ppl = torch.exp(text_nlls.sum() / (ids.size(0) - leap))
            ppls.append(ppl.cpu().numpy().tolist())
    return ppls


def compute_perplexities(texts, model, tokenizer, settings: Settings, device: str) -> list[float]:
    max_length = settings.window if settings.window > 0 else model.config.n_positions
    if settings.ppl_batched and max_length >= 2:
        return batched_global_perplexities(
            texts, model, tokenizer, settings.window, device, settings.leap, settings.ppl_windows_per_batch
        )
    from utils.perplexity import compute_global_perplexity

    return [
        compute_global_perplexity(text, model, tokenizer, settings.window, device=device, leap=settings.leap)
        for text in texts
    ]


# --------------------------------------------------------------------------- #
#                                   Worker                                    #
# --------------------------------------------------------------------------- #

def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(temporary, "w") as file:
        file.write(content)
    os.replace(temporary, path)


@contextmanager
def fixed_walk_order(train_dir: str, order: tuple[str, ...]):
    """Durante il training, os.walk visita le cartelle paziente di train_dir nell'ordine registrato invece che in
    quello del filesystem (os.walk segue la lista dirs, riordinabile sul posto). Gli altri os.walk non cambiano."""
    original_walk = os.walk
    target = os.path.normpath(train_dir)
    rank = {name: index for index, name in enumerate(order)}

    def ordered_walk(top, topdown=True, onerror=None, followlinks=False):
        for root, dirs, files in original_walk(top, topdown, onerror, followlinks):
            if topdown and os.path.normpath(root) == target:
                if set(dirs) != set(rank):
                    raise RuntimeError(f"Patient folders in {root} differ from the recorded order")
                dirs.sort(key=rank.__getitem__)
            yield root, dirs, files

    os.walk = ordered_walk
    try:
        yield
    finally:
        os.walk = original_walk


def load_model(model_dir: Path, device: str):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(str(model_dir))
    return model, tokenizer


def execute_job(job: Job) -> dict[str, object]:
    import torch
    from utils.models_training import train_model

    settings = job.settings
    device = "cuda:0"
    targets = job.missing_targets()
    trained = False

    if targets:
        model = tokenizer = None
        if base.model_dir_is_complete(job.model_dir):
            try:
                model, tokenizer = load_model(job.model_dir, device)
                print(f"Reusing existing model: {job.model_dir}")
            except Exception as exc:
                if is_out_of_memory(exc):
                    raise
                print(f"Existing model is unreadable, retraining: {job.model_dir}")
                shutil.rmtree(job.model_dir)
        if model is None:
            if job.model_dir.exists():
                shutil.rmtree(job.model_dir)
            with fixed_walk_order(job.train_dir, job.train_order):
                train_model(
                    model_type=settings.model_type,
                    model_name=settings.model_name,
                    max_epochs=settings.max_epochs,
                    batch_size=settings.batch_size,
                    train_set_file_path=base.train_dir_arg(Path(job.train_dir)),
                    base_output_dir=job.model_base_dir,
                    loo_folder=job.loo_folder,
                    save_every=settings.save_every,
                    gradient_checkpointing=settings.gradient_checkpointing,
                )
            trained = True
            if not base.model_dir_is_complete(job.model_dir):
                raise RuntimeError(f"Model was not created: {job.model_dir}")
            model, tokenizer = load_model(job.model_dir, device)

        texts = []
        for target in targets:
            with open(target.text_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
        ppls = compute_perplexities(texts, model, tokenizer, settings, device)
        for target, ppl in zip(targets, ppls):
            for output_path in target.output_paths:
                write_text_atomic(Path(output_path), f"{ppl}")

        del model
        torch.cuda.empty_cache()

    if job.model_dir.exists():
        shutil.rmtree(job.model_dir)
    return {"trained": trained, "ppl_texts": len(targets)}


def is_out_of_memory(exc: BaseException) -> bool:
    import torch

    while exc is not None:
        message = str(exc).lower()
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message or "alloc_failed" in message:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def remove_training_tmp(job: Job) -> None:
    # train_model lascia la sua cartella temporanea se il training si interrompe.
    from utils import models_training

    temp_root, _, _ = models_training.temporary_training_paths(job.model_base_dir)
    shutil.rmtree(temp_root, ignore_errors=True)


def worker_main(worker_id: int, conn, memory_gb, log_path: str) -> None:
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    sys.stdout = sys.stderr = log_file

    import torch

    parent_pid = os.getppid()

    def report_memory() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)  # processo principale morto (kill): niente worker orfani
            memory_gb.value = torch.cuda.memory_reserved() / GB
            time.sleep(1.0)

    threading.Thread(target=report_memory, daemon=True).start()

    while True:
        job = conn.recv()
        if job is None:
            break
        print(f"\n===== worker {worker_id} | {datetime.now():%Y-%m-%d %H:%M:%S} | {job.key} =====")
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            info = execute_job(job)
            info["peak_gb"] = torch.cuda.max_memory_reserved() / GB
            info["seconds"] = time.time() - started
            conn.send(("done", job.key, info))
        except Exception as exc:
            details = traceback.format_exc()
            print(details)
            remove_training_tmp(job)
            conn.send(("oom" if is_out_of_memory(exc) else "error", job.key, details))
            break  # dopo un errore il processo si chiude: il job riparte in un processo nuovo


# --------------------------------------------------------------------------- #
#                                  Scheduler                                  #
# --------------------------------------------------------------------------- #

class Worker:
    def __init__(self, ctx, worker_id: int, gpu_index: int, physical_gpu: str, log_path: Path) -> None:
        self.worker_id = worker_id
        self.gpu_index = gpu_index
        self.memory_gb = ctx.RawValue("d", 0.0)
        self.conn, child_conn = ctx.Pipe()
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
        try:
            self.process = ctx.Process(
                target=worker_main,
                args=(worker_id, child_conn, self.memory_gb, str(log_path)),
                daemon=True,
            )
            self.process.start()
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        child_conn.close()
        self.job: Job | None = None
        self.started_at = 0.0

    def stop(self) -> None:
        if self.process.is_alive():
            try:
                self.conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        self.process.join(timeout=60)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GpuLedger:
    """Memoria GPU che ogni run prevede ancora di allocare per i suoi job in corso, per GPU (UUID)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.host = socket.gethostname()
        self.owner = f"{self.host}:{os.getpid()}"

    @contextmanager
    def locked(self):
        with open(self.path.with_name(f"{self.path.name}.lock"), "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _read(self) -> dict[str, dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def _is_live(self, entry: dict, now: float) -> bool:
        if now - entry.get("updated", 0) > GPU_LEDGER_STALE_SECONDS:
            return False
        return entry.get("host") != self.host or pid_alive(int(entry.get("pid", 0)))

    def others(self) -> dict[str, float]:
        """Da chiamare con il lock preso."""
        now = time.time()
        totals: dict[str, float] = {}
        for owner, entry in self._read().items():
            if owner != self.owner and self._is_live(entry, now):
                for gpu, gb in entry.get("growth_gb", {}).items():
                    totals[gpu] = totals.get(gpu, 0.0) + float(gb)
        return totals

    def publish(self, growth_gb: dict[str, float] | None) -> None:
        """Da chiamare con il lock preso; None toglie la voce di questo run."""
        now = time.time()
        data = {owner: entry for owner, entry in self._read().items() if owner != self.owner and self._is_live(entry, now)}
        if growth_gb is not None:
            data[self.owner] = {"host": self.host, "pid": os.getpid(), "updated": now, "growth_gb": growth_gb}
        temporary = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)


def save_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f"{state_path.name}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, state_path)


def mark(state_path: Path, state: dict[str, object], section: str, key: str, status: str, details=None) -> None:
    state.setdefault(section, {})[key] = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
    }
    save_state(state_path, state)


class Scheduler:
    def __init__(self, jobs: list[Job], logger: logging.Logger, state_path: Path, state: dict, workers_log_dir: Path):
        import torch

        self.torch = torch
        self.jobs = jobs
        self.logger = logger
        self.state_path = state_path
        self.state = state
        self.workers_log_dir = workers_log_dir
        self.ctx = mp.get_context("spawn")

        available = torch.cuda.device_count()
        if available == 0:
            raise RuntimeError("No GPU available")
        self.num_gpus = available if NUM_GPUS is None else min(NUM_GPUS, available)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = [item.strip() for item in visible.split(",") if item.strip()] if visible else []
        self.physical_gpus = ids[: self.num_gpus] if ids else [str(index) for index in range(self.num_gpus)]
        self.gpu_uuids = [str(torch.cuda.get_device_properties(index).uuid) for index in range(self.num_gpus)]
        self.ledger = GpuLedger(GPU_LEDGER_PATH)

        self.done = {job.key for job in jobs if not job.missing_targets()}
        self.pending = deque(job for job in jobs if job.key not in self.done)
        self.workers: list[Worker] = []
        self.next_worker_id = 0
        self.peak_gb: float | None = None
        self.parallel_cap = MAX_PARALLEL_JOBS
        self.successes_since_oom = 0
        self.cooldown_until = 0.0
        self.oom_attempts: Counter[str] = Counter()
        self.error_attempts: Counter[str] = Counter()
        self.failed: str | None = None
        self.classified: set[int] = set()
        self.fold_metrics: dict[int, dict[str, object]] = {}
        self.started_at = time.time()
        self.completed_this_run = 0
        self.last_status = 0.0
        self.last_block_log = 0.0

    # ---------------------------- memory ---------------------------- #

    def busy(self) -> list[Worker]:
        return [worker for worker in self.workers if worker.job is not None]

    def job_peak_estimate(self) -> float:
        return self.peak_gb * 1.1 if self.peak_gb is not None else FIRST_JOB_MIN_FREE_GB

    def own_growth(self) -> dict[str, float]:
        # I job appena partiti non hanno ancora allocato: si riserva per ognuno il picco stimato.
        growth = {uuid: 0.0 for uuid in self.gpu_uuids}
        for worker in self.busy():
            growth[self.gpu_uuids[worker.gpu_index]] += max(0.0, self.job_peak_estimate() - worker.memory_gb.value)
        return growth

    def gpu_headroom(self, gpu_index: int, others: dict[str, float]) -> tuple[float, float, float]:
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(gpu_index)
        uuid = self.gpu_uuids[gpu_index]
        return free_bytes / GB, total_bytes / GB, self.own_growth()[uuid] + others.get(uuid, 0.0)

    def publish_reservations(self) -> None:
        with self.ledger.locked():
            self.ledger.publish(self.own_growth())

    def block(self, reason: str) -> None:
        if time.time() - self.last_block_log >= 60:
            self.logger.info("Waiting before next job: %s", reason)
            self.last_block_log = time.time()

    def dispatch(self) -> None:
        while self.pending and self.failed is None:
            busy = self.busy()
            if time.time() < self.cooldown_until:
                return self.block(f"cooldown after OOM until {datetime.fromtimestamp(self.cooldown_until):%H:%M:%S}")
            if len(busy) >= self.parallel_cap:
                return
            if self.peak_gb is None and busy:
                return self.block("first job runs alone to measure its GPU memory peak")

            disk_free = shutil.disk_usage(models_root()).free / GB
            disk_need = MIN_FREE_DISK_GB + DISK_PER_JOB_GB * (len(busy) + 1)
            if disk_free < disk_need:
                return self.block(f"disk free {disk_free:.1f} GB < {disk_need:.1f} GB")

            # Decisione e prenotazione sotto lo stesso lock, cosi' due run non scelgono la stessa memoria libera.
            with self.ledger.locked():
                others = self.ledger.others()
                best = None
                for gpu_index in range(self.num_gpus):
                    free, total, reserved = self.gpu_headroom(gpu_index, others)
                    idle = next((w for w in self.workers if w.job is None and w.gpu_index == gpu_index), None)
                    need = self.job_peak_estimate() + GPU_MEMORY_MARGIN_GB + (0.0 if idle else WORKER_CONTEXT_GB)
                    headroom = free - reserved - need
                    if best is None or headroom > best[0]:
                        best = (headroom, gpu_index, idle, free, total, reserved, need)
                headroom, gpu_index, idle, free, total, reserved, need = best
                if headroom < 0:
                    return self.block(
                        f"GPU {gpu_index} free {free:.1f}/{total:.1f} GB, reserved for starting jobs "
                        f"(this run + other runs) {reserved:.1f} GB, needed {need:.1f} GB"
                    )

                worker = idle or self.spawn_worker(gpu_index)
                job = self.pending.popleft()
                worker.memory_gb.value = 0.0  # il worker lo aggiorna entro 1 s; finche' no, si prenota tutto il picco
                worker.conn.send(job)
                worker.job = job
                worker.started_at = time.time()
                self.ledger.publish(self.own_growth())
            self.last_block_log = 0.0
            mark(self.state_path, self.state, "jobs", job.key, "in_progress", {"worker": worker.worker_id})
            self.logger.info(
                "Started %s on worker %d (GPU %d) | running=%d free=%.1f GB reserved=%.1f GB peak_est=%.1f GB",
                job.key, worker.worker_id, gpu_index, len(busy) + 1, free, reserved, self.job_peak_estimate(),
            )

    def spawn_worker(self, gpu_index: int) -> Worker:
        worker_id = self.next_worker_id
        self.next_worker_id += 1
        log_path = self.workers_log_dir / f"worker_{worker_id}.log"
        worker = Worker(self.ctx, worker_id, gpu_index, self.physical_gpus[gpu_index], log_path)
        self.workers.append(worker)
        self.logger.info("Spawned worker %d on GPU %d (log: %s)", worker_id, gpu_index, log_path)
        return worker

    # ---------------------------- events ---------------------------- #

    def retire(self, worker: Worker) -> None:
        worker.stop()
        self.workers.remove(worker)

    def handle(self, worker: Worker, kind: str, payload) -> None:
        job = worker.job
        worker.job = None
        running_with = len(self.busy()) + 1

        if kind == "done":
            self.done.add(job.key)
            self.completed_this_run += 1
            if payload["trained"]:
                self.peak_gb = max(self.peak_gb or 0.0, float(payload["peak_gb"]))
            self.successes_since_oom += 1
            if self.parallel_cap < MAX_PARALLEL_JOBS and self.successes_since_oom >= SUCCESSES_TO_RAISE_CAP:
                self.parallel_cap += 1
                self.successes_since_oom = 0
                self.logger.info("Parallel cap raised to %d", self.parallel_cap)
            mark(self.state_path, self.state, "jobs", job.key, "completed", payload)
            self.logger.info(
                "Completed %s in %.0f s | peak=%.2f GB | done %d/%d",
                job.key, payload["seconds"], payload["peak_gb"], len(self.done), len(self.jobs),
            )
            self.classify_ready_folds()
            return

        self.retire(worker)
        if kind == "oom":
            self.oom_attempts[job.key] += 1
            self.parallel_cap = max(1, running_with - 1)
            self.successes_since_oom = 0
            self.cooldown_until = time.time() + OOM_COOLDOWN_SECONDS
            self.logger.warning(
                "CUDA OOM on %s (attempt %d, %d jobs running) -> requeued, parallel cap %d",
                job.key, self.oom_attempts[job.key], running_with, self.parallel_cap,
            )
            if self.oom_attempts[job.key] > MAX_OOM_RETRIES:
                self.failed = f"{job.key}: too many OOM retries"
        else:
            self.error_attempts[job.key] += 1
            self.logger.error("Job %s failed (%s, attempt %d):\n%s", job.key, kind, self.error_attempts[job.key], payload)
            if self.error_attempts[job.key] > MAX_ERROR_RETRIES:
                self.failed = f"{job.key}: {kind}"
        mark(self.state_path, self.state, "jobs", job.key, "failed" if self.failed else "requeued", {"error": str(payload)[-2000:]})
        self.pending.appendleft(job)

    def poll(self) -> None:
        busy = self.busy()
        if not busy:
            time.sleep(POLL_SECONDS)
            return
        ready = wait_connections([worker.conn for worker in busy], timeout=POLL_SECONDS)
        for worker in busy:
            if worker.conn in ready:
                try:
                    kind, _, payload = worker.conn.recv()
                except (EOFError, OSError):
                    worker.process.join(timeout=10)
                    kind, payload = "crash", f"worker exited with code {worker.process.exitcode}"
                self.handle(worker, kind, payload)
            elif not worker.process.is_alive():
                self.handle(worker, "crash", f"worker exited with code {worker.process.exitcode}")

    def log_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_status < STATUS_LOG_SECONDS:
            return
        self.last_status = now
        gpu_info = []
        for gpu_index in range(self.num_gpus):
            free, total, _ = self.gpu_headroom(gpu_index, {})
            gpu_info.append(f"GPU{gpu_index} free {free:.1f}/{total:.1f} GB")
        remaining = len(self.jobs) - len(self.done)
        eta = ""
        if self.completed_this_run:
            seconds = (now - self.started_at) / self.completed_this_run * remaining
            eta = f" | ETA ~{seconds / 3600:.1f} h"
        self.logger.info(
            "Status | done %d/%d | running %d | pending %d | cap %d | peak_est %s | %s%s",
            len(self.done), len(self.jobs), len(self.busy()), len(self.pending), self.parallel_cap,
            f"{self.job_peak_estimate():.1f} GB" if self.peak_gb else "n/a", ", ".join(gpu_info), eta,
        )

    # ------------------------- classification ------------------------ #

    def classify_ready_folds(self) -> None:
        for fold in FOLDS:
            if fold in self.classified or not fold_job_keys(self.jobs, fold) <= self.done:
                continue
            self.classified.add(fold)
            try:
                self.fold_metrics[fold] = run_fold_classification(fold, self.logger)
                mark(self.state_path, self.state, "folds", fold_name(fold), "completed", self.fold_metrics[fold])
            except Exception as exc:
                self.logger.error("Classification of %s failed: %s", fold_name(fold), exc)
                mark(self.state_path, self.state, "folds", fold_name(fold), "failed", {"error": str(exc)})

    # ------------------------------ run ------------------------------ #

    def run(self) -> bool:
        self.logger.info(
            "Jobs: %d total, %d already complete, %d to run | GPUs used: %s",
            len(self.jobs), len(self.done), len(self.pending), self.physical_gpus,
        )
        self.classify_ready_folds()
        try:
            while True:
                self.dispatch()
                if not self.busy() and (not self.pending or self.failed):
                    break
                self.poll()
                self.publish_reservations()
                self.log_status()
        finally:
            for worker in list(self.workers):
                if worker.job is not None:
                    self.logger.warning("Stopping worker %d while running %s", worker.worker_id, worker.job.key)
                    worker.process.terminate()
                worker.stop()
            self.workers.clear()
            with self.ledger.locked():
                self.ledger.publish(None)
        self.log_status(force=True)
        if self.failed:
            self.logger.error("Run stopped: %s", self.failed)
            return False
        return True


# --------------------------------------------------------------------------- #
#                               Classification                                #
# --------------------------------------------------------------------------- #

def run_fold_classification(fold: int, logger: logging.Logger) -> dict[str, object]:
    # Come run_experiment.run_delta_leave_one_out_classification, con la cartella risultati del fold.
    control_output_dir = fold_ppl_dir(fold, CONTROL_CLASS)
    disease_output_dir = fold_ppl_dir(fold, DISEASE_CLASS)
    control_model_name = ppl_model_name(CONTROL_CLASS)
    disease_model_name = ppl_model_name(DISEASE_CLASS)

    control_records = base.load_class_d_scores(
        class_name=CONTROL_CLASS,
        output_dir=control_output_dir,
        control_model_name=control_model_name,
        disease_model_name=disease_model_name,
        control_score_is_loo=True,
        disease_score_is_loo=False,
    )
    disease_records = base.load_class_d_scores(
        class_name=DISEASE_CLASS,
        output_dir=disease_output_dir,
        control_model_name=control_model_name,
        disease_model_name=disease_model_name,
        control_score_is_loo=False,
        disease_score_is_loo=True,
    )
    rows = base.classify_with_leave_one_out_means(
        control_records=control_records,
        disease_records=disease_records,
    )
    metrics = base.classification_metrics(rows)

    results_dir = results_dir_for(fold_name(fold))
    output_csv = results_dir / "leave_one_out_predictions_from_ppl_table.csv"
    output_json = results_dir / "leave_one_out_metrics.json"
    output_plot_png = results_dir / "dscore_classification_plot.png"
    output_plot_pdf = results_dir / "dscore_classification_plot.pdf"

    base.write_classification_csv(rows, output_csv)
    plot_stats = base.create_d_score_plot(rows, output_plot_png, output_plot_pdf)
    metrics["d_score_reference"] = plot_stats
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "%s classification | total=%d correct=%d accuracy=%.4f | %s",
        fold_name(fold), metrics["total"], metrics["correct"], metrics["accuracy"], results_dir,
    )
    return {**metrics, "results_dir": str(results_dir)}


def write_folds_summary(fold_metrics: dict[int, dict[str, object]], logger: logging.Logger) -> None:
    rows = []
    for fold in sorted(fold_metrics):
        metrics = fold_metrics[fold]
        by_label = metrics["by_label"]
        reference = metrics["d_score_reference"]
        rows.append(
            {
                "fold": fold,
                "dataset": fold_name(fold),
                "total": metrics["total"],
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                f"{CONTROL_CLASS}_accuracy": by_label[CONTROL_CLASS]["accuracy"],
                f"{DISEASE_CLASS}_accuracy": by_label[DISEASE_CLASS]["accuracy"],
                "cn_mean_d": reference["cn_mean_d"],
                "ad_mean_d": reference["ad_mean_d"],
                "threshold_d": reference["threshold_d"],
            }
        )

    summary: dict[str, object] = {"folds": [row["fold"] for row in rows]}
    for column in ("accuracy", f"{CONTROL_CLASS}_accuracy", f"{DISEASE_CLASS}_accuracy"):
        values = [float(row[column]) for row in rows]
        summary[column] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    confusion: dict[str, dict[str, int]] = {}
    for fold in sorted(fold_metrics):
        for true_label, predicted in fold_metrics[fold]["confusion_matrix"].items():
            for predicted_label, count in predicted.items():
                confusion.setdefault(true_label, {}).setdefault(predicted_label, 0)
                confusion[true_label][predicted_label] += count
    summary["pooled_confusion_matrix"] = confusion

    results_dir = results_dir_for(FOLDS_RUN_NAME)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "folds_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "folds_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "Folds summary | accuracy mean=%.4f std=%.4f | %s",
        summary["accuracy"]["mean"], summary["accuracy"]["std"], results_dir,
    )


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #

def remove_leftover_models(jobs: list[Job], logger: logging.Logger) -> None:
    for job in jobs:
        if not job.missing_targets() and job.model_dir.exists():
            logger.info("Removing model of completed job %s: %s", job.key, job.model_dir)
            shutil.rmtree(job.model_dir)


def remove_stale_training_tmp(jobs: list[Job], logger: logging.Logger) -> None:
    # Cartelle temporanee di train_model (<parent>/.training_tmp/<timestamp>_pid<N>) di processi non piu' vivi.
    for tmp_parent in sorted({Path(job.model_base_dir).parent / ".training_tmp" for job in jobs}):
        if not tmp_parent.is_dir():
            continue
        for run_dir in tmp_parent.iterdir():
            pid = run_dir.name.rsplit("_pid", 1)[-1]
            if pid.isdigit() and not pid_alive(int(pid)):
                logger.info("Removing stale training temp dir: %s", run_dir)
                shutil.rmtree(run_dir, ignore_errors=True)
        try:
            tmp_parent.rmdir()
        except OSError:
            pass


def raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def remove_empty_model_dirs(jobs: list[Job]) -> None:
    for final_dir in sorted({job.final_dir for job in jobs}):
        try:
            final_dir.rmdir()
        except OSError:
            pass


def print_plan(jobs: list[Job]) -> None:
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        group = job.key.rsplit("/", 1)[0] if "loo" in job.key else job.key
        groups.setdefault(group, []).append(job)
    recorded = json.loads(WALK_ORDER_PATH.read_text(encoding="utf-8"))
    print(f"Epochs: {MAX_EPOCHS} | models root: {models_root()}")
    print(f"Training order: {WALK_ORDER_PATH.name} (recorded on {recorded['recorded_on']} at {recorded['recorded_at']})")
    differs = [key for key, order in recorded["dirs"].items() if filesystem_walk_order(PROJECT_ROOT / key) != order]
    print(f"  this filesystem lists {len(differs)}/{len(recorded['dirs'])} training dirs in a different order (imposed anyway)")
    for group, group_jobs in groups.items():
        complete = sum(1 for job in group_jobs if not job.missing_targets())
        outputs = sum(len(target.output_paths) for job in group_jobs for target in job.targets)
        print(f"  {group:<22} jobs {complete:>3}/{len(group_jobs):<3} complete | PPL files {outputs}")
    for fold in FOLDS:
        print(f"  {fold_name(fold)}: PPL -> {fold_ppl_dir(fold, CONTROL_CLASS).parents[2]}")
        print(f"  {' ' * len(fold_name(fold))}  results -> {results_dir_for(fold_name(fold))}")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "--record-walk-order" in sys.argv[1:]:
        return record_walk_order()
    walk_order = load_walk_order()
    validate_folds(walk_order)
    jobs = build_jobs(current_settings(), walk_order)
    if "--dry-run" in sys.argv[1:]:
        print_plan(jobs)
        return 0

    logger = setup_logging(timestamp)
    logger.info("Starting Anchise/Nestor fold experiment | folds=%s epochs=%d", list(FOLDS), MAX_EPOCHS)

    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_file = open(root / f"run_folds_{run_tag()}.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another run_experiment_fold with %d epochs is already running", MAX_EPOCHS)
        return 1

    state_path = root / f"run_state_folds_{run_tag()}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    workers_log_dir = PROJECT_ROOT / "logs" / f"anchise_nestor_folds_{MAX_EPOCHS}ep_{timestamp}_workers"
    workers_log_dir.mkdir(parents=True, exist_ok=True)

    # kill (SIGTERM) come Ctrl+C: si fermano i worker invece di lasciarli orfani.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    try:
        remove_stale_training_tmp(jobs, logger)
        remove_leftover_models(jobs, logger)
        scheduler = Scheduler(jobs, logger, state_path, state, workers_log_dir)
        completed = scheduler.run()
        if not completed:
            return 1

        fold_metrics = {}
        for fold in FOLDS:
            fold_metrics[fold] = scheduler.fold_metrics.get(fold) or run_fold_classification(fold, logger)
        write_folds_summary(fold_metrics, logger)
        mark(state_path, state, "run", "summary", "completed", {"results_dir": str(results_dir_for(FOLDS_RUN_NAME))})
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted; completed PPL files are kept and the next run resumes from them")
        return 1
    except Exception as exc:
        logger.error("Error during fold experiment: %s\n%s", exc, traceback.format_exc())
        mark(state_path, state, "run", "last_error", "failed", {"error": str(exc)})
        return 1
    finally:
        remove_stale_training_tmp(jobs, logger)  # dei worker fermati durante un training
        remove_empty_model_dirs(jobs)


if __name__ == "__main__":
    sys.exit(main())
