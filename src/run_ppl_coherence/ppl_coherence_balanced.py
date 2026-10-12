"""Coerenza con la perplexity di GePpeTto sul dataset bilanciato (anchise_nestor_balanced_completi_cleaned, 28 AD e 28 CN).

Stesso esperimento di ppl_coherence_folds.py, da cui importa costruzione delle coppie, PPL, regola, metriche, job,
worker e scheduler; cambia il dataset (uno solo invece dei 5 fold):

1. Dataset. Da anchise_nestor_balanced_completi_cleaned crea resources/data/input/anchise_nestor_coherence_balanced
   nella forma di anchise_nestor_coherence (anchise/<id>/<id>_adj.txt e <id>_nadj.txt per gli AD, nestor/... per i
   CN), con le stesse coppie adj e nadj dei fold. I file gia' presenti non vengono riscritti.
2. Modelli CN: il globale (PPL delle coppie dei 28 AD) e i 28 leave-one-out (PPL delle coppie del CN escluso), con
   train_model, gli stessi argomenti e l'ordine dei pazienti registrato per questo dataset in walk_order.json (quello
   di run_experiment_optimized). Ogni modello e' cancellato appena le sue PPL sono scritte.
   Con --epochs 0 nessun training: un solo GePpeTto pre-addestrato calcola le PPL di tutti i pazienti.
3. PPL delle coppie ("battuta_i\\nbattuta_j", finestra 20, leap 0), regola (CN se coerenza media adj > nadj; chi ha
   meno di 3 battute non e' classificabile e conta come errore) e metriche (Accuracy, Pd, Rd, F1d, Pc, Rc, F1c, HM,
   AUC) come nei fold, sui 56 pazienti. Le predizioni per paziente servono per i test statistici.

Uso (da src/): HF_HUB_OFFLINE=1 python -m run_ppl_coherence.ppl_coherence_balanced --epochs E
               [--dry-run | --build-dataset | --classify-only]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import fcntl
import json
import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_ppl_coherence import ppl_coherence_folds as pc  # noqa: E402
from run_ppl_coherence import ppl_coherence_folds_base as pcb  # noqa: E402
from runnestoranchise import run_experiment_optimized as ro  # noqa: E402

rf = pc.rf
AD, CN = pc.AD, pc.CN
PROJECT_ROOT = pc.PROJECT_ROOT
INPUT_ROOT = PROJECT_ROOT / "resources" / "data" / "input"
DATASET_NAME = "anchise_nestor_balanced_completi_cleaned"
COHERENCE_DATASET = "anchise_nestor_coherence_balanced"
DEFAULT_EPOCHS = 10


def source_dir(class_name: str) -> Path:
    return INPUT_ROOT / DATASET_NAME / "train" / class_name / rf.TEXT_FOLDER


def coherence_patient_dir(class_name: str, pid: str) -> Path:
    return INPUT_ROOT / COHERENCE_DATASET / pc.CLASS_FOLDERS[class_name] / pid


def members() -> list[tuple[str, str]]:
    return [(class_name, pid) for class_name in (AD, CN) for pid in rf.patient_ids(source_dir(class_name))]


def run_paths(epochs: int) -> pc.RunPaths:
    model_name = f"{rf.model_base_name(CN)}_{epochs}ep"  # cn_manual_16b_<E>ep, come nei fold
    data = PROJECT_ROOT / "resources" / "data"
    tag = f"{rf.MODEL_TYPE}_{model_name}_w{rf.WINDOW}_l{rf.LEAP}"
    return pc.RunPaths(
        tag=tag,
        models_root=data / "output" / "models" / COHERENCE_DATASET / rf.MODEL_TYPE,
        ppl_dir=data / f"output_{rf.MODEL_TYPE}" / f"{COHERENCE_DATASET}_w{rf.WINDOW}_l{rf.LEAP}" / model_name,
        results_dir=data / "results" / "ppl_coherence_balanced" / tag,
        log_name=f"ppl_coherence_balanced_{epochs}ep",
    )


def training_order() -> tuple[str, ...]:
    # Ordine dei CN registrato in walk_order.json, imposto a train_model come in run_experiment_optimized.
    key = rf.relative_key(source_dir(CN))
    walk_order = ro.load_walk_order()
    if key not in walk_order:
        raise ValueError(f"{key} is missing from {ro.WALK_ORDER_PATH.name}")
    if sorted(walk_order[key]) != sorted(rf.patient_ids(source_dir(CN))):
        raise ValueError(f"Patients in {source_dir(CN)} differ from those recorded in {ro.WALK_ORDER_PATH.name}")
    return tuple(walk_order[key])


# --------------------------------------------------------------------------- #
#                                   Dataset                                   #
# --------------------------------------------------------------------------- #

def build_coherence_dataset() -> tuple[int, int]:
    # Come ppl_coherence_folds.build_coherence_datasets, su questo dataset.
    written = unchanged = 0
    for class_name, pid in members():
        utts = pc.utterances(source_dir(class_name) / pid / rf.TEST_FILE_NAME)
        if any("\t" in utt for utt in utts):
            raise ValueError(f"Tab inside an utterance of {source_dir(class_name) / pid}")
        for kind, pairs in pc.pair_indices(len(utts)).items():
            path = pc.pair_file(coherence_patient_dir(class_name, pid), kind)
            content = "".join(f"{utts[i]}\t{utts[j]}\n" for i, j in pairs)
            if path.exists():
                if path.read_text(encoding="utf-8") != content:
                    raise RuntimeError(f"{path} exists with different content; remove {COHERENCE_DATASET} to rebuild it")
                unchanged += 1
            else:
                rf.write_text_atomic(path, content)
                written += 1
    return written, unchanged


def pair_counts() -> dict[str, dict[str, int]]:
    counts = {class_name: {kind: 0 for kind in pc.PAIR_KINDS} for class_name in (AD, CN)}
    for class_name, pid in members():
        for kind in pc.PAIR_KINDS:
            counts[class_name][kind] += len(pc.read_pairs(pc.pair_file(coherence_patient_dir(class_name, pid), kind)))
    return counts


# --------------------------------------------------------------------------- #
#                                  Training                                   #
# --------------------------------------------------------------------------- #

def build_jobs(settings: rf.Settings, order: tuple[str, ...], paths: pc.RunPaths) -> list[rf.Job]:
    # Come ppl_coherence_folds.build_jobs, con i 28 AD e i 28 CN di questo dataset.
    control_dir = source_dir(CN)

    def target(class_name: str, pid: str) -> rf.PplTarget:
        return rf.PplTarget(
            text_path=str(coherence_patient_dir(class_name, pid)),
            output_paths=(str(pc.ppl_file(paths, class_name, pid)),),
        )

    global_job = rf.Job(
        key="cn_global",
        train_dir=str(control_dir),
        train_order=order,
        model_base_dir=str(paths.models_root / rf.model_base_name(CN)),
        loo_folder=None,
        targets=tuple(target(AD, pid) for pid in rf.patient_ids(source_dir(AD))),
        settings=settings,
    )
    loo_jobs = [
        rf.Job(
            key=f"cn_loo/{pid}",
            train_dir=str(control_dir),
            train_order=order,
            model_base_dir=str(paths.models_root / "leave_one_out" / rf.model_base_name(CN)),
            loo_folder=pid,
            targets=(target(CN, pid),),
            settings=settings,
        )
        for pid in rf.patient_ids(control_dir)
    ]
    # Primo un LOO (breve): lo scheduler fa girare da solo il primo job per misurarne il picco di memoria GPU.
    return loo_jobs[:1] + [global_job] + loo_jobs[1:]


class BalancedScheduler(pc.CoherenceScheduler):
    """Lo scheduler dei fold, con i resoconti man mano sui 28 AD e 28 CN di questo dataset."""

    def log_job(self, job: rf.Job) -> None:
        if job.loo_folder is not None:
            super().log_job(job)  # una riga per il CN escluso
            return
        keys = [f"{AD}/{pid}" for pid in rf.patient_ids(source_dir(AD))]
        wrong = [key for key in keys if self.summaries[key]["predicted_label"] != AD]
        self.logger.info(
            "Modello CN globale | AD corretti %d/%d (%.1f%%) | sbagliati: %s",
            len(keys) - len(wrong), len(keys), 100 * (len(keys) - len(wrong)) / len(keys),
            ", ".join(f"{key} ({self.summaries[key]['predicted_label']})" for key in wrong) or "nessuno",
        )

    def log_running(self) -> None:
        log_running(self.summaries, self.logger)


def log_running(summaries: dict[str, dict], logger: logging.Logger) -> None:
    parts = []
    for class_name in (AD, CN):
        keys = [f"{class_name}/{pid}" for pid in rf.patient_ids(source_dir(class_name))]
        done = [key for key in keys if key in summaries]
        ok = sum(1 for key in done if summaries[key]["predicted_label"] == class_name)
        parts.append(f"{class_name.upper()} corretti {ok}/{len(done)} (di {len(keys)})")
    done = [key for key in summaries]
    ok = sum(1 for key in done if summaries[key]["predicted_label"] == key.split("/")[0])
    parts.append(f"accuratezza finora {ok}/{len(done)} ({100 * ok / len(done):.1f}%)")
    logger.info("Parziale | %s", " | ".join(parts))


def run_pretrained(paths: pc.RunPaths, settings: rf.Settings, logger: logging.Logger) -> None:
    """0 epoche: il CN globale e i 28 LOO sono tutti GePpeTto pre-addestrato, un solo modello per tutti i pazienti."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    summaries = {}
    todo = []
    for class_name, pid in members():
        path = pc.ppl_file(paths, class_name, pid)
        if path.exists():
            summaries[f"{class_name}/{pid}"] = pc.summarize(pc.read_patient_ppls(path))
        else:
            todo.append((class_name, pid))
    logger.info("GePpeTto 0 epochs (not fine-tuned) | %d patients already done, %d to do", len(summaries), len(todo))
    if not todo:
        return

    device, free_gb = pcb.pick_device()
    logger.info("Model %s on %s (free %.1f GB)", settings.model_name, device, free_gb)
    model = GPT2LMHeadModel.from_pretrained(settings.model_name).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(settings.model_name)
    started_run = time.time()
    for index, (class_name, pid) in enumerate(todo, start=1):
        started = time.time()
        patient_dir = coherence_patient_dir(class_name, pid)
        pairs = {kind: pc.read_pairs(pc.pair_file(patient_dir, kind)) for kind in pc.PAIR_KINDS}
        texts = [first + pc.PAIR_SEPARATOR + second for kind in pc.PAIR_KINDS for first, second in pairs[kind]]
        ppls = pc.compute_pair_perplexities(texts, model, tokenizer, settings, device)
        rf.write_text_atomic(pc.ppl_file(paths, class_name, pid), pc.ppl_table(pairs, ppls))
        key = f"{class_name}/{pid}"
        summaries[key] = pc.summarize({"adj": ppls[: len(pairs["adj"])], "nadj": ppls[len(pairs["adj"]) :]})
        eta_min = (time.time() - started_run) / index * (len(todo) - index) / 60
        logger.info("[%d/%d] %s | %.0f s | mancano ~%.0f min", index, len(todo), pc.summary_line(key, summaries[key]),
                    time.time() - started, eta_min)
        log_running(summaries, logger)
    del model
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#                                  Risultati                                  #
# --------------------------------------------------------------------------- #

def classify(paths: pc.RunPaths, logger: logging.Logger) -> dict[str, float] | None:
    rows, missing = [], []
    for class_name, pid in members():
        path = pc.ppl_file(paths, class_name, pid)
        if not path.exists():
            missing.append(f"{class_name}/{pid}")
            continue
        summary = pc.summarize(pc.read_patient_ppls(path))
        rows.append({"patient_key": f"{class_name}/{pid}", "patient_id": pid, "true_label": class_name,
                     **summary, "correct": summary["predicted_label"] == class_name})
    if missing:
        logger.error("PPL missing for %d patients (e.g. %s): run without --classify-only first", len(missing), ", ".join(missing[:5]))
        return None

    pc.count_unclassified_as_errors(rows)  # non classificabili: errore in tutte le metriche, come nei fold
    metrics = pc.fold_metrics(rows)
    results = paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    pc.write_predictions(results / "predictions.csv", rows, pc.PREDICTION_FIELDS)
    with open(results / "metrics.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Dataset"] + pc.METRIC_NAMES + pc.COUNT_NAMES)
        writer.writerow([DATASET_NAME] + [f"{metrics[n]:.10g}" for n in pc.METRIC_NAMES] + [metrics[n] for n in pc.COUNT_NAMES])
    (results / "metrics.json").write_text(json.dumps({"dataset": DATASET_NAME, **metrics}, indent=2), encoding="utf-8")
    for row in rows:
        if not row["classified"]:
            logger.warning("%s not classifiable (adj pairs %d, nadj pairs %d): counted as an error, predicted %s",
                           row["patient_key"], row["n_adj"], row["n_nadj"], row["predicted_label"])
    logger.info("===== %s | %s | NC %d", DATASET_NAME, pc.bert.metrics_line(metrics), metrics["NC"])
    logger.info("Results in %s", results)
    return metrics


def write_settings(paths: pc.RunPaths, settings: rf.Settings) -> None:
    pc.write_settings(paths, settings, smoke=False)
    path = paths.results_dir / "settings.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    recorded = json.loads(ro.WALK_ORDER_PATH.read_text(encoding="utf-8")).get("recorded", {})
    key = rf.relative_key(source_dir(CN))
    content.pop("std", None)
    content.update({
        "dataset": DATASET_NAME,
        "datasets": f"resources/data/input/{COHERENCE_DATASET}",
        "training_order": f"{ro.WALK_ORDER_PATH.name} ({recorded.get(key, 'recorded')})",
        "not_classified": "patients without adj or nadj pairs (classified = False) are in every metric as errors: "
        "predicted with the wrong class; score_ad = dataset minimum - 1 if AD, dataset maximum + 1 if CN",
    })
    if settings.max_epochs == 0:
        content.pop("training_order", None)
        content["training"] = f"none: {settings.model_name} as is (0 epochs), one model for all patients"
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def print_plan(jobs: list[rf.Job], paths: pc.RunPaths, settings: rf.Settings) -> None:
    counts = pair_counts()
    print(f"Dataset: {DATASET_NAME} -> {INPUT_ROOT / COHERENCE_DATASET}")
    for class_name in (AD, CN):
        c = counts[class_name]
        print(f"  {class_name}: {len(rf.patient_ids(source_dir(class_name)))} patients, {c['adj']} adj + {c['nadj']} nadj pairs")
    print(f"Epochs: {settings.max_epochs} | {settings.model_name} | batch {settings.batch_size} | PPL window {settings.window}, leap {settings.leap}")
    if settings.max_epochs == 0:
        done = sum(1 for class_name, pid in members() if pc.ppl_file(paths, class_name, pid).exists())
        print(f"  no training: one pretrained model for all patients | PPL done for {done}/{len(members())} patients")
    else:
        print(f"  training order: {ro.WALK_ORDER_PATH.name}")
        for group in ("cn_global", "cn_loo"):
            group_jobs = [job for job in jobs if job.key.split("/")[0] == group]
            complete = sum(1 for job in group_jobs if not job.missing_targets())
            print(f"  {group:<10} jobs {complete:>2}/{len(group_jobs):<2} complete | {sum(len(job.targets) for job in group_jobs)} patients")
        print(f"  models   -> {paths.models_root} (deleted after their PPL)")
    print(f"  PPL      -> {paths.ppl_dir}")
    print(f"  results  -> {paths.results_dir}")


# --------------------------------------------------------------------------- #
#                                     Main                                    #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="0 = pretrained GePpeTto, no training")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="build the dataset and print the plan")
    mode.add_argument("--build-dataset", action="store_true", help="only build the coherence dataset")
    mode.add_argument("--classify-only", action="store_true", help="only recompute the results from the PPL files")
    args = parser.parse_args()

    paths = run_paths(args.epochs)
    settings = dataclasses.replace(rf.current_settings(), max_epochs=args.epochs, save_every=args.epochs)
    order = training_order()
    written, unchanged = build_coherence_dataset()
    print(f"Coherence dataset {COHERENCE_DATASET}: {written} pair files written, {unchanged} already present and identical")
    if args.build_dataset:
        return 0
    jobs = build_jobs(settings, order, paths) if args.epochs > 0 else []
    if args.dry_run:
        print_plan(jobs, paths, settings)
        return 0

    logger, workers_log_dir = pc.setup_logging(paths, datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.classify_only:
        return 0 if classify(paths, logger) is not None else 1

    logger.info("PPL coherence on %s | %s", DATASET_NAME, paths.tag)
    write_settings(paths, settings)
    paths.models_root.mkdir(parents=True, exist_ok=True)
    lock_file = open(paths.models_root / f"run_coherence_{paths.tag}.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another ppl_coherence_balanced run with %d epochs is already running", args.epochs)
        return 1

    signal.signal(signal.SIGTERM, rf.raise_keyboard_interrupt)
    if args.epochs == 0:
        try:
            run_pretrained(paths, settings, logger)
        except KeyboardInterrupt:
            logger.warning("Interrupted; PPL files written so far are kept and the next run resumes from them")
            return 1
        return 0 if classify(paths, logger) is not None else 1

    state_path = paths.models_root / f"run_state_coherence_{paths.tag}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    workers_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Worker logs (training and per-patient PPL): %s", workers_log_dir)

    # Come in ppl_coherence_folds: lo scheduler di run_experiment_fold misura il disco sulla cartella dei modelli di
    # questo run e lancia i worker che eseguono ppl_coherence_folds.execute_coherence_job.
    rf.models_root = lambda: paths.models_root
    rf.worker_main = pc.coherence_worker_main
    try:
        rf.remove_stale_training_tmp(jobs, logger)
        rf.remove_leftover_models(jobs, logger)
        scheduler = BalancedScheduler(jobs, logger, state_path, state, workers_log_dir, smoke=False)
        if not scheduler.run():
            return 1
        if classify(paths, logger) is None:
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
