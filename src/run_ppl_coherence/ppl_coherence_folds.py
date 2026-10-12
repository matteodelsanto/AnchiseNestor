"""Coerenza con la perplexity di GePpeTto sui 5 fold Anchise/Nestor (resources/data/input/anchise_nestor_fold_k).

Stesso principio di ppl_coherence.py, sui fold:

1. Dataset. Da ogni fold crea resources/data/input/anchise_nestor_coherence_fold_k nella forma di
   anchise_nestor_coherence: anchise/<id>/<id>_adj.txt e <id>_nadj.txt per gli AD, nestor/... per i CN.
   Le battute sono le righe di test_text.txt; adj = battute consecutive (i, i+1), nadj = tutte le coppie (i, j)
   con j >= i+2; una coppia per riga, "battuta_i<TAB>battuta_j". I file gia' presenti non vengono riscritti.
2. Modelli. Solo CN, allenati una volta perche' Nestor e' identico in tutti i fold: il globale (PPL delle coppie
   dei 140 AD) e i 28 leave-one-out (PPL delle coppie del CN escluso). Training, ordine dei pazienti
   (fold_walk_order.json), scheduler GPU e worker sono quelli di run_experiment_fold, importati; di nuovo c'e' solo
   il corpo del job (execute_coherence_job). Ogni modello e' cancellato appena le sue PPL sono scritte.
3. PPL. Ogni coppia diventa "battuta_i\\nbattuta_j" (a capo, come tra le battute dei testi di training) e la sua PPL
   e' quella dell'esperimento GPT sui fold (run_experiment_fold.compute_perplexities, finestra 20, leap 0), calcolata
   con la stessa aritmetica da fast_global_perplexities: finestre distinte calcolate una volta, vocabolario proiettato
   solo dove serve, batch grandi (PPL_FAST = False usa la funzione di run_experiment_fold).
   Le PPL di tutte le coppie restano su disco, un file per paziente.
4. Regola di ppl_coherence.py. Coerenza di una coppia = 1 - PPL; per paziente la media sulle adj e sulle nadj:
   CN se media adj > media nadj, altrimenti AD. Chi non ha coppie adj o nadj (meno di 3 battute, es. ad/78) non e'
   classificabile e non e' escluso da nessuna metrica: conta come errore (classe sbagliata e, per l'AUC, punteggio
   peggiore del fold); la colonna NC dice quanti sono.
5. Risultati, per fold (28 AD del fold + 28 CN) e media/DS (ddof=1) sui fold, negli stessi file della baseline BERT:
   Accuracy, Pd, Rd, F1d (AD), Pc, Rc, F1c (CN), HM(Accuracy, F1d, F1c), AUC sul punteggio coerenza nadj - adj,
   matrice di confusione. Le predizioni per paziente (anche tutte in un file) servono per i test statistici.

Uso (da src/): HF_HUB_OFFLINE=1 python -m run_ppl_coherence.ppl_coherence_folds [--epochs 10]
               [--dry-run | --build-dataset | --classify-only] [--smoke-test]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import fcntl
import json
import logging
import shutil
import signal
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runnestoranchise import run_bert_loo_folds as bert  # noqa: E402
from runnestoranchise import run_experiment_fold as rf  # noqa: E402

base = rf.base

PROJECT_ROOT = rf.PROJECT_ROOT
FOLDS = rf.FOLDS
AD, CN = rf.DISEASE_CLASS, rf.CONTROL_CLASS
DEFAULT_EPOCHS = 10
COHERENCE_DATASET_PREFIX = "anchise_nestor_coherence_fold_"
CLASS_FOLDERS = {AD: "anchise", CN: "nestor"}  # come FOLDER_LABELS di ppl_coherence.py
PAIR_KINDS = ("adj", "nadj")
PAIR_SEPARATOR = "\n"  # come tra le battute dei testi di training
NOT_CLASSIFIED = "nc"
SMOKE_PATIENTS = 2  # --smoke-test: 1 epoca, 2 CN LOO e 2 AD per fold, cartelle a parte
PPL_FAST = True  # False = run_experiment_fold.compute_perplexities (stessa aritmetica, piu' lenta)
FIRST_WINDOWS_PER_BATCH = 256  # prime finestre, logit su tutte le posizioni: 256 x 20 x 30000 x 4 B = ~600 MB
NEXT_WINDOWS_PER_BATCH = 2048  # finestre successive, logit solo sul penultimo token; i batch si dimezzano se la GPU e' piena

METRIC_NAMES = ["Accuracy", "Pd", "Rd", "F1d", "Pc", "Rc", "F1c", "HM", "AUC"]
COUNT_NAMES = ["AD→AD", "AD→CN", "CN→AD", "CN→CN", "NC"]
PREDICTION_FIELDS = [
    "patient_key", "patient_id", "true_label", "n_adj", "n_nadj", "ppl_adj_mean", "ppl_nadj_mean",
    "coherence_adj", "coherence_nadj", "score_ad", "predicted_label", "classified", "correct",
]


@dataclasses.dataclass(frozen=True)
class RunPaths:
    tag: str
    models_root: Path
    ppl_dir: Path
    results_dir: Path
    log_name: str


def run_paths(epochs: int, smoke: bool) -> RunPaths:
    model_name = f"{rf.model_base_name(CN)}_{epochs}ep"  # cn_manual_16b_<E>ep, come nei file PPL di run_experiment
    suffix = "_smoke_test" if smoke else ""
    data = PROJECT_ROOT / "resources" / "data"
    tag = f"{rf.MODEL_TYPE}_{model_name}_w{rf.WINDOW}_l{rf.LEAP}{suffix}"
    return RunPaths(
        tag=tag,
        models_root=data / "output" / "models" / f"anchise_nestor_coherence_folds{suffix}" / rf.MODEL_TYPE,
        ppl_dir=data / f"output_{rf.MODEL_TYPE}" / f"anchise_nestor_coherence_folds{suffix}_w{rf.WINDOW}_l{rf.LEAP}" / model_name,
        results_dir=data / "results" / "ppl_coherence_folds" / tag,
        log_name=f"ppl_coherence_folds_{epochs}ep{suffix}",
    )


def coherence_patient_dir(fold: int, class_name: str, pid: str) -> Path:
    root = PROJECT_ROOT / "resources" / "data" / "input" / f"{COHERENCE_DATASET_PREFIX}{fold}"
    return root / CLASS_FOLDERS[class_name] / pid


def pair_file(patient_dir: Path, kind: str) -> Path:
    return patient_dir / f"{patient_dir.name}_{kind}.txt"


def ppl_file(paths: RunPaths, class_name: str, pid: str) -> Path:
    return paths.ppl_dir / class_name / f"{pid}.tsv"


def target_key(target: rf.PplTarget) -> str:
    path = Path(target.output_paths[0])
    return f"{path.parent.name}/{path.stem}"


def fold_members(fold: int, smoke: bool) -> list[tuple[str, str]]:
    limit = SMOKE_PATIENTS if smoke else None
    return [(class_name, pid) for class_name in (AD, CN) for pid in rf.patient_ids(rf.fold_class_dir(fold, class_name))[:limit]]


# --------------------------------------------------------------------------- #
#                                   Dataset                                   #
# --------------------------------------------------------------------------- #

def utterances(text_path: Path) -> list[str]:
    lines = (line.strip() for line in text_path.read_text(encoding="utf-8").split("\n"))
    return [line for line in lines if line]


def pair_indices(n_utterances: int) -> dict[str, list[tuple[int, int]]]:
    # Come build_pairs_from_utterances di run_coherence_roberta.
    return {
        "adj": [(i, i + 1) for i in range(n_utterances - 1)],
        "nadj": [(i, j) for i in range(n_utterances) for j in range(i + 2, n_utterances)],
    }


def build_coherence_datasets() -> tuple[int, int]:
    written = unchanged = 0
    for fold in FOLDS:
        for class_name in (AD, CN):
            source_dir = rf.fold_class_dir(fold, class_name)
            for pid in rf.patient_ids(source_dir):
                utts = utterances(source_dir / pid / rf.TEST_FILE_NAME)
                if any("\t" in utt for utt in utts):
                    raise ValueError(f"Tab inside an utterance of {source_dir / pid}")
                for kind, pairs in pair_indices(len(utts)).items():
                    path = pair_file(coherence_patient_dir(fold, class_name, pid), kind)
                    content = "".join(f"{utts[i]}\t{utts[j]}\n" for i, j in pairs)
                    if path.exists():
                        if path.read_text(encoding="utf-8") != content:
                            raise RuntimeError(f"{path} exists with different content; remove the coherence folders to rebuild them")
                        unchanged += 1
                    else:
                        rf.write_text_atomic(path, content)
                        written += 1
    return written, unchanged


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line:
            first, second = line.split("\t")
            pairs.append((first, second))
    return pairs


# --------------------------------------------------------------------------- #
#                              Perplexity e regola                            #
# --------------------------------------------------------------------------- #

def ppl_table(pairs: dict[str, list[tuple[str, str]]], ppls: list[float]) -> str:
    # Gli indici delle battute si ricavano dall'ordine delle righe nei file adj/nadj.
    indices = pair_indices(len(pairs["adj"]) + 1)
    if any(len(indices[kind]) != len(pairs[kind]) for kind in PAIR_KINDS):
        raise ValueError("Pair files do not match the adj/nadj construction")
    values = iter(ppls)
    lines = ["kind\tu1\tu2\tppl"]
    for kind in PAIR_KINDS:
        lines.extend(f"{kind}\t{i}\t{j}\t{next(values)!r}" for i, j in indices[kind])
    return "\n".join(lines) + "\n"


def run_in_batches(items: list, batch_size: int, step) -> None:
    # step(chunk, start) su blocchi di items; se la GPU e' piena il blocco si dimezza e si riprova.
    import torch

    start = 0
    while start < len(items):
        chunk = items[start : start + batch_size]
        try:
            step(chunk, start)
        except Exception as exc:
            if not rf.is_out_of_memory(exc) or batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"GPU memory full: windows per batch lowered to {batch_size}", flush=True)
            continue
        start += len(chunk)


def fast_global_perplexities(texts: list[str], model, tokenizer, settings: rf.Settings, device: str) -> list[float]:
    """Stessa aritmetica di run_experiment_fold.batched_global_perplexities (e di utils.perplexity.compute_global_perplexity):
    finestre di `window` token con stride 1; la prima finestra pesa la media delle sue NLL x la sua lunghezza, ognuna
    delle altre la NLL del suo ultimo token; PPL = exp(somma dal leap-esimo valore / (token - leap)). Cambia solo come
    si calcola:
    - ogni finestra distinta si calcola una volta (le coppie (i, j) e (i, k) iniziano con la stessa battuta i);
    - per le finestre dopo la prima si proietta sul vocabolario solo il penultimo token, l'unico che serve
      (transformer e lm_head come in GPT2LMHeadModel.forward);
    - batch grandi."""
    import torch
    import torch.nn.functional as F

    window = settings.window if settings.window > 0 else model.config.n_positions
    leap = settings.leap
    encoded = tokenizer(texts)["input_ids"] if texts else []

    first_windows: dict[tuple[int, ...], int] = {}
    next_windows: dict[tuple[int, ...], int] = {}
    text_windows = []
    for ids in encoded:
        if len(ids) < 2:
            raise ValueError("Cannot compute the perplexity of a text shorter than two tokens")
        first = first_windows.setdefault(tuple(ids[:window]), len(first_windows))
        following = [
            next_windows.setdefault(tuple(ids[begin : begin + window]), len(next_windows))
            for begin in range(1, len(ids) - window + 1)
        ]
        text_windows.append((first, following))

    first_nll = torch.empty(len(first_windows), dtype=torch.float32, device=device)
    next_nll = torch.empty(len(next_windows), dtype=torch.float32, device=device)

    def first_step(chunk, _start):
        input_ids = torch.tensor([key for key, _ in chunk], device=device)
        logits = model(input_ids).logits.float()
        length = input_ids.size(1)
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1), reduction="none"
        ).view(len(chunk), length - 1)
        first_nll[torch.tensor([index for _, index in chunk], device=device)] = losses.mean(dim=1) * length

    def next_step(chunk, start):
        input_ids = torch.tensor(chunk, device=device)
        hidden = model.transformer(input_ids, use_cache=False)[0]
        logits = model.lm_head(hidden[:, -2]).float()
        next_nll[start : start + len(chunk)] = F.cross_entropy(logits, input_ids[:, -1], reduction="none")

    with torch.inference_mode():
        by_length: dict[int, list[tuple[tuple[int, ...], int]]] = {}
        for key, index in first_windows.items():
            by_length.setdefault(len(key), []).append((key, index))
        for items in by_length.values():
            run_in_batches(items, FIRST_WINDOWS_PER_BATCH, first_step)
        run_in_batches(list(next_windows), NEXT_WINDOWS_PER_BATCH, next_step)

    first_nll, next_nll = first_nll.cpu(), next_nll.cpu()
    ppls = []
    for ids, (first, following) in zip(encoded, text_windows):
        nlls = torch.cat([first_nll[first : first + 1], next_nll[torch.tensor(following, dtype=torch.long)]])
        ppls.append(torch.exp(nlls[leap:].sum() / (len(ids) - leap)).numpy().tolist())
    return ppls


def compute_pair_perplexities(texts: list[str], model, tokenizer, settings: rf.Settings, device: str) -> list[float]:
    if PPL_FAST:
        return fast_global_perplexities(texts, model, tokenizer, settings, device)
    return rf.compute_perplexities(texts, model, tokenizer, settings, device)


def read_patient_ppls(path: Path) -> dict[str, list[float]]:
    ppl_by_kind: dict[str, list[float]] = {kind: [] for kind in PAIR_KINDS}
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            ppl_by_kind[row["kind"]].append(float(row["ppl"]))
    return ppl_by_kind


def summarize(ppl_by_kind: dict[str, list[float]]) -> dict[str, object]:
    """Regola di ppl_coherence.py: coerenza = 1 - PPL, media per tipo di coppia; CN se adj > nadj, altrimenti AD."""
    mean = lambda values: statistics.mean(values) if values else None
    coherence = {kind: mean([1 - float(ppl) for ppl in ppl_by_kind[kind]]) for kind in PAIR_KINDS}
    if coherence["adj"] is None or coherence["nadj"] is None:
        predicted = NOT_CLASSIFIED
    elif coherence["adj"] > coherence["nadj"]:
        predicted = CN
    else:
        predicted = AD
    return {
        "n_adj": len(ppl_by_kind["adj"]),
        "n_nadj": len(ppl_by_kind["nadj"]),
        "ppl_adj_mean": mean(ppl_by_kind["adj"]),
        "ppl_nadj_mean": mean(ppl_by_kind["nadj"]),
        "coherence_adj": coherence["adj"],
        "coherence_nadj": coherence["nadj"],
        "score_ad": None if predicted == NOT_CLASSIFIED else coherence["nadj"] - coherence["adj"],  # >= 0 -> AD
        "predicted_label": predicted,
    }


def fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def summary_line(key: str, s: dict[str, object]) -> str:
    true = key.split("/")[0]
    return (
        f"{key}: coppie adj {s['n_adj']}, nadj {s['n_nadj']} | PPL media adj {fmt(s['ppl_adj_mean'])}, "
        f"nadj {fmt(s['ppl_nadj_mean'])} -> {str(s['predicted_label']).upper()} "
        f"{'ok' if s['predicted_label'] == true else 'ERR'}"
    )


# --------------------------------------------------------------------------- #
#                                    Worker                                   #
# --------------------------------------------------------------------------- #

def execute_coherence_job(job: rf.Job) -> dict[str, object]:
    """run_experiment_fold.execute_job con le PPL delle coppie di battute di ogni paziente al posto del testo intero."""
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
                model, tokenizer = rf.load_model(job.model_dir, device)
                print(f"Reusing existing model: {job.model_dir}")
            except Exception as exc:
                if rf.is_out_of_memory(exc):
                    raise
                print(f"Existing model is unreadable, retraining: {job.model_dir}")
                shutil.rmtree(job.model_dir)
        if model is None:
            if job.model_dir.exists():
                shutil.rmtree(job.model_dir)
            with rf.fixed_walk_order(job.train_dir, job.train_order):
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
            model, tokenizer = rf.load_model(job.model_dir, device)

        print(f"PPL of the pairs of {len(targets)} patients with {job.model_dir}", flush=True)
        for index, target in enumerate(targets, start=1):
            started = time.time()
            patient_dir = Path(target.text_path)
            pairs = {kind: read_pairs(pair_file(patient_dir, kind)) for kind in PAIR_KINDS}
            texts = [first + PAIR_SEPARATOR + second for kind in PAIR_KINDS for first, second in pairs[kind]]
            ppls = compute_pair_perplexities(texts, model, tokenizer, settings, device)
            for output_path in target.output_paths:
                rf.write_text_atomic(Path(output_path), ppl_table(pairs, ppls))
            summary = summarize({"adj": ppls[: len(pairs["adj"])], "nadj": ppls[len(pairs["adj"]) :]})
            print(f"[{index}/{len(targets)}] {summary_line(target_key(target), summary)} | {time.time() - started:.0f} s", flush=True)

        del model
        torch.cuda.empty_cache()

    if job.model_dir.exists():
        shutil.rmtree(job.model_dir)
    return {"trained": trained, "ppl_texts": len(targets)}


def coherence_worker_main(worker_id: int, conn, memory_gb, log_path: str) -> None:
    # Il processo worker di run_experiment_fold, che qui esegue execute_coherence_job al posto di execute_job.
    rf.execute_job = execute_coherence_job
    rf.worker_main(worker_id, conn, memory_gb, log_path)


# --------------------------------------------------------------------------- #
#                                  Scheduler                                  #
# --------------------------------------------------------------------------- #

def build_jobs(settings: rf.Settings, walk_order: dict[str, list[str]], paths: RunPaths, smoke: bool) -> list[rf.Job]:
    # Come i job cn di run_experiment_fold.build_jobs, con le coppie di battute come bersagli.
    control_dir = rf.fold_class_dir(FOLDS[0], CN)
    control_order = tuple(walk_order[rf.relative_key(control_dir)])
    limit = SMOKE_PATIENTS if smoke else None

    def target(fold: int, class_name: str, pid: str) -> rf.PplTarget:
        return rf.PplTarget(
            text_path=str(coherence_patient_dir(fold, class_name, pid)),
            output_paths=(str(ppl_file(paths, class_name, pid)),),
        )

    global_job = rf.Job(
        key="cn_global",
        train_dir=str(control_dir),
        train_order=control_order,
        model_base_dir=str(paths.models_root / rf.model_base_name(CN)),
        loo_folder=None,
        targets=tuple(
            target(fold, AD, pid) for fold in FOLDS for pid in rf.patient_ids(rf.fold_class_dir(fold, AD))[:limit]
        ),
        settings=settings,
    )
    loo_jobs = [
        rf.Job(
            key=f"cn_loo/{pid}",
            train_dir=str(control_dir),
            train_order=control_order,
            model_base_dir=str(paths.models_root / "leave_one_out" / rf.model_base_name(CN)),
            loo_folder=pid,
            targets=(target(FOLDS[0], CN, pid),),  # Nestor e' uguale in tutti i fold
            settings=settings,
        )
        for pid in rf.patient_ids(control_dir)[:limit]
    ]
    # Primo un LOO (breve): lo scheduler fa girare da solo il primo job per misurarne il picco di memoria GPU.
    return loo_jobs[:1] + [global_job] + loo_jobs[1:]


class CoherenceScheduler(rf.Scheduler):
    """Lo scheduler di run_experiment_fold; al posto della classificazione per fold stampa i risultati man mano."""

    def __init__(self, *args, smoke: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.smoke = smoke
        self.reported: set[str] = set()
        self.summaries: dict[str, dict[str, object]] = {}

    def classify_ready_folds(self) -> None:
        # Scheduler la chiama all'avvio e dopo ogni job completato.
        new_jobs = [job for job in self.jobs if job.key in self.done and job.key not in self.reported]
        try:
            for job in new_jobs:
                self.reported.add(job.key)
                for target in job.targets:
                    self.summaries[target_key(target)] = summarize(read_patient_ppls(Path(target.output_paths[0])))
                self.log_job(job)
            if new_jobs:
                self.log_running()
        except Exception as exc:
            self.logger.error("Progress report failed: %s\n%s", exc, traceback.format_exc())

    def log_job(self, job: rf.Job) -> None:
        if job.loo_folder is not None:
            key = target_key(job.targets[0])
            self.logger.info("Modello CN LOO | %s", summary_line(key, self.summaries[key]))
            return
        for fold in FOLDS:
            keys = [f"{c}/{pid}" for c, pid in fold_members(fold, self.smoke) if c == AD]
            wrong = [key for key in keys if self.summaries[key]["predicted_label"] != AD]
            self.logger.info(
                "Modello CN globale | fold %d: AD corretti %d/%d (%.1f%%) | sbagliati: %s",
                fold, len(keys) - len(wrong), len(keys), 100 * (len(keys) - len(wrong)) / len(keys),
                ", ".join(f"{key} ({self.summaries[key]['predicted_label']})" for key in wrong) or "nessuno",
            )

    def log_running(self) -> None:
        cn_keys = [f"{CN}/{pid}" for c, pid in fold_members(FOLDS[0], self.smoke) if c == CN]
        cn_done = [key for key in cn_keys if key in self.summaries]
        cn_ok = sum(1 for key in cn_done if self.summaries[key]["predicted_label"] == CN)
        parts = [f"CN corretti {cn_ok}/{len(cn_done)} (di {len(cn_keys)})"]
        for fold in FOLDS:
            done = [f"{c}/{pid}" for c, pid in fold_members(fold, self.smoke) if f"{c}/{pid}" in self.summaries]
            ok = sum(1 for key in done if self.summaries[key]["predicted_label"] == key.split("/")[0])
            if done:
                parts.append(f"fold {fold} {ok}/{len(done)} ({100 * ok / len(done):.1f}%)")
        self.logger.info("Parziale | %s", " | ".join(parts))


# --------------------------------------------------------------------------- #
#                                  Risultati                                  #
# --------------------------------------------------------------------------- #

def count_unclassified_as_errors(rows: list[dict[str, object]]) -> None:
    """Chi non e' classificabile entra in tutte le metriche come errore: predetto con la classe sbagliata e, per l'AUC,
    con il punteggio peggiore del fold (sotto tutti i CN se e' AD, sopra tutti gli AD se e' CN)."""
    scores = [row["score_ad"] for row in rows if row["score_ad"] is not None]
    for row in rows:
        row["classified"] = row["predicted_label"] != NOT_CLASSIFIED
        if not row["classified"]:
            true_ad = row["true_label"] == AD
            row["predicted_label"] = CN if true_ad else AD
            row["score_ad"] = min(scores, default=0.0) - 1 if true_ad else max(scores, default=0.0) + 1
            row["correct"] = False


def fold_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    # Come run_bert_loo_folds.fold_metrics, sul punteggio score_ad; NC = non classificabili, gia' contati come errori.
    from scipy.stats import hmean
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    y = [row["true_label"] for row in rows]
    p = [row["predicted_label"] for row in rows]
    acc = sum(a == b for a, b in zip(y, p)) / len(y)
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=[AD, CN], zero_division=0)
    hm = hmean([acc, F[0], F[1]]) if min(acc, F[0], F[1]) > 0 else 0.0
    labels = [1 if label == AD else 0 for label in y]
    auc = roc_auc_score(labels, [row["score_ad"] for row in rows]) if 0 < sum(labels) < len(labels) else float("nan")
    cm = {f"{t.upper()}→{q.upper()}": sum(1 for a, b in zip(y, p) if a == t and b == q) for t in (AD, CN) for q in (AD, CN)}
    nc = sum(1 for row in rows if not row["classified"])
    return {"Accuracy": acc, "Pd": P[0], "Rd": R[0], "F1d": F[0], "Pc": P[1], "Rc": R[1], "F1c": F[1], "HM": hm, "AUC": auc, **cm, "NC": nc}


def write_summaries(out_dir: Path, per_fold: dict[int, dict], logger: logging.Logger) -> None:
    # Stessi file e formato di run_bert_loo_folds.write_summaries, con le colonne dei non classificati.
    with open(out_dir / "fold_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Fold"] + METRIC_NAMES + COUNT_NAMES)
        for fold, m in sorted(per_fold.items()):
            w.writerow([fold] + [f"{m[n]:.10g}" for n in METRIC_NAMES] + [m[n] for n in COUNT_NAMES])
    if len(per_fold) < 2:
        return
    summary = {n: {"mean": statistics.mean(m[n] for m in per_fold.values()),
                   "std": statistics.stdev(m[n] for m in per_fold.values())} for n in METRIC_NAMES + COUNT_NAMES}
    with open(out_dir / "folds_aggregate.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Metric", "Avg", "Std"])
        for n in METRIC_NAMES + COUNT_NAMES:
            w.writerow([n, f"{summary[n]['mean']:.5f}", f"{summary[n]['std']:.5f}"])
    (out_dir / "folds_aggregate.json").write_text(json.dumps({"folds": sorted(per_fold), **summary}, indent=2), encoding="utf-8")
    logger.info("===== Media ± DS (ddof=1) su %d fold =====", len(per_fold))
    for n in METRIC_NAMES:
        logger.info("  %-8s %.1f ± %.1f", n, 100 * summary[n]["mean"], 100 * summary[n]["std"])


def csv_value(value) -> object:
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else value


def write_predictions(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for row in rows:
            w.writerow([csv_value(row[field]) for field in fields])


def classify_folds(paths: RunPaths, smoke: bool, logger: logging.Logger) -> dict[int, dict] | None:
    rows_by_fold: dict[int, list[dict[str, object]]] = {}
    missing = []
    for fold in FOLDS:
        rows_by_fold[fold] = []
        for class_name, pid in fold_members(fold, smoke):
            path = ppl_file(paths, class_name, pid)
            if not path.exists():
                missing.append(f"{class_name}/{pid}")
                continue
            summary = summarize(read_patient_ppls(path))
            rows_by_fold[fold].append({
                "fold": fold, "patient_key": f"{class_name}/{pid}", "patient_id": pid, "true_label": class_name,
                **summary, "correct": summary["predicted_label"] == class_name,
            })
    if missing:
        logger.error("PPL missing for %d patients (e.g. %s): run without --classify-only first", len(set(missing)), ", ".join(sorted(set(missing))[:5]))
        return None

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    per_fold = {}
    for fold, rows in rows_by_fold.items():
        count_unclassified_as_errors(rows)
        write_predictions(paths.results_dir / f"predictions_fold_{fold}.csv", rows, PREDICTION_FIELDS)
        per_fold[fold] = fold_metrics(rows)
        logger.info("===== Fold %d | %s | NC %d", fold, bert.metrics_line(per_fold[fold]), per_fold[fold]["NC"])
        for row in rows:
            if not row["classified"]:
                logger.warning("Fold %d: %s not classifiable (adj pairs %d, nadj pairs %d): counted as an error, "
                               "predicted %s, AUC score %.4g", fold, row["patient_key"], row["n_adj"], row["n_nadj"],
                               row["predicted_label"], row["score_ad"])
    all_rows = [row for rows in rows_by_fold.values() for row in rows]
    write_predictions(paths.results_dir / "predictions_all_folds.csv", all_rows, ["fold"] + PREDICTION_FIELDS)
    write_summaries(paths.results_dir, per_fold, logger)
    logger.info("Results in %s", paths.results_dir)
    return per_fold


# --------------------------------------------------------------------------- #
#                                     Main                                    #
# --------------------------------------------------------------------------- #

def setup_logging(paths: RunPaths, timestamp: str) -> tuple[logging.Logger, Path]:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / f"{paths.log_name}_{timestamp}.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("ppl_coherence_folds"), log_dir / f"{paths.log_name}_{timestamp}_workers"


def write_settings(paths: RunPaths, settings: rf.Settings, smoke: bool) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    recorded = json.loads(rf.WALK_ORDER_PATH.read_text(encoding="utf-8"))
    content = {
        **dataclasses.asdict(settings),
        "training": "utils.models_training.train_model via run_experiment_fold (cn global + 28 cn LOO)",
        "training_order": f"{rf.WALK_ORDER_PATH.name} (recorded on {recorded['recorded_on']} at {recorded['recorded_at']})",
        "ppl": "run_experiment_fold.compute_perplexities on each pair",
        "pair_separator": PAIR_SEPARATOR,
        "rule": "coherence = 1 - PPL; CN if mean coherence adj > mean coherence nadj, else AD",
        "not_classified": "patients without adj or nadj pairs (classified = False) are in every metric as errors: "
        "predicted with the wrong class; score_ad = fold minimum - 1 if AD, fold maximum + 1 if CN",
        "score_ad": "coherence_nadj - coherence_adj (AD if >= 0); used for AUC",
        "std": "statistics.stdev (ddof=1) over folds",
        "datasets": f"resources/data/input/{COHERENCE_DATASET_PREFIX}<k>",
        "ppl_dir": str(paths.ppl_dir.relative_to(PROJECT_ROOT)),
        "smoke_test": smoke,
    }
    (paths.results_dir / "settings.json").write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def print_plan(jobs: list[rf.Job], paths: RunPaths, settings: rf.Settings) -> None:
    recorded = json.loads(rf.WALK_ORDER_PATH.read_text(encoding="utf-8"))
    print(f"Epochs: {settings.max_epochs} | {settings.model_name} | batch {settings.batch_size} | PPL window {settings.window}, leap {settings.leap}")
    print(f"Training order: {rf.WALK_ORDER_PATH.name} (recorded on {recorded['recorded_on']} at {recorded['recorded_at']})")
    for group in ("cn_global", "cn_loo"):
        group_jobs = [job for job in jobs if job.key.split("/")[0] == group]
        complete = sum(1 for job in group_jobs if not job.missing_targets())
        patients = sum(len(job.targets) for job in group_jobs)
        pairs = sum(
            len(read_pairs(pair_file(Path(target.text_path), kind)))
            for job in group_jobs for target in job.targets for kind in PAIR_KINDS
        )
        print(f"  {group:<10} jobs {complete:>2}/{len(group_jobs):<2} complete | {patients} patients, {pairs} pairs")
    print(f"  datasets -> {PROJECT_ROOT / 'resources' / 'data' / 'input'}/{COHERENCE_DATASET_PREFIX}1..{FOLDS[-1]}")
    print(f"  models   -> {paths.models_root} (deleted after their PPL)")
    print(f"  PPL      -> {paths.ppl_dir}")
    print(f"  results  -> {paths.results_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="build the datasets and print the plan")
    mode.add_argument("--build-dataset", action="store_true", help="only build the coherence datasets")
    mode.add_argument("--classify-only", action="store_true", help="only recompute the results from the PPL files")
    parser.add_argument("--smoke-test", action="store_true", help=f"1 epoch, {SMOKE_PATIENTS} CN LOO and {SMOKE_PATIENTS} AD per fold")
    args = parser.parse_args()

    epochs = 1 if args.smoke_test else args.epochs
    paths = run_paths(epochs, args.smoke_test)
    settings = dataclasses.replace(rf.current_settings(), max_epochs=epochs, save_every=epochs)
    walk_order = rf.load_walk_order()
    rf.validate_folds(walk_order)  # anche: testi e ordine cn identici in tutti i fold

    written, unchanged = build_coherence_datasets()
    print(f"Coherence datasets: {written} pair files written, {unchanged} already present and identical")
    if args.build_dataset:
        return 0
    jobs = build_jobs(settings, walk_order, paths, args.smoke_test)
    if args.dry_run:
        print_plan(jobs, paths, settings)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, workers_log_dir = setup_logging(paths, timestamp)
    if args.classify_only:
        return 0 if classify_folds(paths, args.smoke_test, logger) is not None else 1

    logger.info("PPL coherence on folds %s | %s", list(FOLDS), paths.tag)
    write_settings(paths, settings, args.smoke_test)
    paths.models_root.mkdir(parents=True, exist_ok=True)
    lock_file = open(paths.models_root / f"run_coherence_{paths.tag}.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another ppl_coherence_folds run with the same settings is already running")
        return 1

    state_path = paths.models_root / f"run_state_coherence_{paths.tag}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    workers_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Worker logs (training and per-patient PPL): %s", workers_log_dir)

    # Lo scheduler di run_experiment_fold misura il disco libero sulla sua cartella dei modelli e lancia i worker con
    # la sua worker_main: qui puntano alla cartella e al worker di questo run.
    rf.models_root = lambda: paths.models_root
    rf.worker_main = coherence_worker_main

    signal.signal(signal.SIGTERM, rf.raise_keyboard_interrupt)
    try:
        rf.remove_stale_training_tmp(jobs, logger)
        rf.remove_leftover_models(jobs, logger)
        scheduler = CoherenceScheduler(jobs, logger, state_path, state, workers_log_dir, smoke=args.smoke_test)
        if not scheduler.run():
            return 1
        if classify_folds(paths, args.smoke_test, logger) is None:
            return 1
        rf.mark(state_path, state, "run", "summary", "completed", {"results_dir": str(paths.results_dir)})
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted; completed PPL files are kept and the next run resumes from them")
        return 1
    except Exception as exc:
        logger.error("Error during the coherence run: %s\n%s", exc, traceback.format_exc())
        rf.mark(state_path, state, "run", "last_error", "failed", {"error": str(exc)})
        return 1
    finally:
        rf.remove_stale_training_tmp(jobs, logger)  # dei worker fermati durante un training
        rf.remove_empty_model_dirs(jobs)


if __name__ == "__main__":
    sys.exit(main())
