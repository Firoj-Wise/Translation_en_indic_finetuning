"""
run_pipeline.py — Single entry point for the Translation pipeline.

Usage:
    python run_pipeline.py --config configs/default.yaml
    python run_pipeline.py --config configs/sample_run.yaml
    python run_pipeline.py --config configs/default.yaml --override training.num_epochs=3

Stages (configurable via ``pipeline.stages``):
    1. ingestion      — Load data from HuggingFace or local files
    2. validation     — Filter bad pairs, deduplicate
    3. preprocessing  — Normalize, split, tokenize
    4. training       — LoRA finetuning with W&B logging
    5. evaluation     — Per-direction BLEU & chrF++ on locked test set
"""

import argparse
import copy
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
import pandas as pd

# ── Logging first ────────────────────────────────────────────
from utils.logging.setup import setup_logger

logger = setup_logger("pipeline", log_file="pipeline.log")


# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge ``override`` into ``base``."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and merge YAML configs.

    If the config has a ``_base_`` key, the base config is loaded
    first and the current file is merged on top.
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle inheritance via _base_
    base_name = config.pop("_base_", None)
    if base_name:
        base_path = path.parent / base_name
        base_config = load_config(str(base_path))
        config = _deep_merge(base_config, config)

    return config


def apply_overrides(config: Dict[str, Any], overrides: list) -> Dict:
    """
    Apply ``key=value`` overrides to the config.

    Supports dotted keys: ``training.num_epochs=3``
    """
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Ignoring malformed override: {override}")
            continue

        key, value = override.split("=", 1)
        keys = key.split(".")

        # Auto-cast value
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"

        # Set nested key
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        logger.info(f"Override: {key} = {value}")

    return config


# ─────────────────────────────────────────────────────────────
# Tracker factory
# ─────────────────────────────────────────────────────────────

def init_tracker(config: Dict[str, Any]):
    """Instantiate the experiment tracker from config."""
    backend = config.get("tracking", {}).get("backend", "wandb")

    if backend == "wandb":
        from pipeline.benchmarking.tracker.wandb_tracker import WandBTracker
        tracker = WandBTracker()
    else:
        from pipeline.benchmarking.tracker.wrapped_tracker import NoOpTracker
        tracker = NoOpTracker()

    track_cfg = config.get("tracking", {})
    tracker.init_run(
        project=track_cfg.get("project", "indictrans2-finetuning"),
        run_name=track_cfg.get("run_name", "indictrans2-finetune"),
        config=config,
    )
    return tracker


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IndicTrans2 Translation Finetuning Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/default.yaml)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Key=value overrides (e.g. training.num_epochs=3)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────
    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    stages = config["pipeline"].get(
        "stages",
        ["ingestion", "validation", "preprocessing", "training", "evaluation"],
    )
    seed = config["pipeline"].get("seed", 42)

    logger.info(f"Pipeline stages: {stages}")
    logger.info(f"Seed: {seed}")

    # ── Environment & reproducibility ─────────────────────────
    from utils.common.seed import set_seed
    from utils.common.environment import log_environment_info

    set_seed(seed)
    env_info = log_environment_info(logger)

    # ── Tracker ───────────────────────────────────────────────
    tracker = init_tracker(config)
    tracker.log_config(config)

    # ── Variables that persist across stages ───────────────────
    df = None
    validation_report = None
    train_dataset = None
    val_dataset = None
    test_df = None
    split_stats = None
    model = None
    tokenizer = None
    run_meta = None
    eval_results = None
    training_start = None
    training_duration = 0.0

    # ==========================================================
    # STAGE 1: INGESTION
    # ==========================================================
    if "ingestion" in stages:
        logger.info("=== Stage 1: Ingestion ===")
        from pipeline.ingestion.loader import load_data
        df = load_data(config)
        logger.info(f"Ingested {len(df):,} rows")

    # ==========================================================
    # STAGE 2: VALIDATION
    # ==========================================================
    if "validation" in stages:
        logger.info("=== Stage 2: Validation ===")
        if df is None:
            raise RuntimeError(
                "Validation requires ingestion to run first."
            )
        from pipeline.validation.validator import validate
        df, validation_report = validate(df, config)

        # Log to tracker
        tracker.log_artifact(
            "validation_report",
            validation_report.to_dict(),
            artifact_type="validation",
        )

        # Save report
        ver_cfg = config.get("versioning", {})
        if ver_cfg.get("enabled", True):
            report_dir = ver_cfg.get("output_dir", "./runs")
            validation_report.save(
                os.path.join(report_dir, "validation_report.json")
            )

    # ==========================================================
    # STAGE 3: PREPROCESSING
    # ==========================================================
    if "preprocessing" in stages:
        logger.info("=== Stage 3: Preprocessing ===")
        if df is None:
            raise RuntimeError(
                "Preprocessing requires ingestion + validation."
            )

        from pipeline.model.loading.indictrans import load_tokenizer
        from pipeline.preprocessing.processor import process

        tokenizer = load_tokenizer(config)
        train_dataset, val_dataset, test_df, split_stats = process(
            df, tokenizer, config
        )

        tracker.log_artifact(
            "split_statistics",
            split_stats.to_dict(),
            artifact_type="preprocessing",
        )

    # ==========================================================
    # VERSIONING (before training)
    # ==========================================================
    ver_cfg = config.get("versioning", {})
    if ver_cfg.get("enabled", True) and df is not None:
        from versioning.version_metadata import create_run_metadata
        run_meta = create_run_metadata(config, df, env_info)
        tracker.log_artifact(
            "run_metadata",
            run_meta.to_dict(),
            artifact_type="versioning",
        )

    # ==========================================================
    # STAGE 4: TRAINING
    # ==========================================================
    if "training" in stages:
        logger.info("=== Stage 4: Training ===")
        if train_dataset is None or val_dataset is None:
            raise RuntimeError(
                "Training requires preprocessing to run first."
            )

        from pipeline.model.loading.indictrans import load_model
        from pipeline.training.training_args import build_training_args
        from pipeline.training.data_collator import build_data_collator
        from pipeline.training.trainer import RobustLoRATrainer
        from pipeline.training.callbacks import (
            VersioningCallback,
            GPUMonitorCallback,
        )
        from pipeline.evaluation.evaluate.compute_metrics import (
            make_compute_metrics,
        )
        from transformers.trainer_utils import get_last_checkpoint

        model = load_model(config)
        training_args = build_training_args(config)
        data_collator = build_data_collator(tokenizer)
        compute_metrics_fn = make_compute_metrics(tokenizer)

        # Build callbacks
        callbacks = [GPUMonitorCallback(log_interval=50)]
        if run_meta:
            callbacks.append(VersioningCallback(run_meta.to_dict()))

        trainer = RobustLoRATrainer(
            training_config=config.get("training", {}),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
        )

        # Check for existing checkpoints to resume
        last_checkpoint = None
        output_dir = config["training"].get("output_dir", "./checkpoints")
        if os.path.isdir(output_dir):
            last_checkpoint = get_last_checkpoint(output_dir)

        logger.info("Starting training ...")
        training_start = time.time()

        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        training_duration = time.time() - training_start
        logger.info(
            f"Training complete in {training_duration / 60:.1f} minutes"
        )

        # ── Merge LoRA ────────────────────────────────────────
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            from pipeline.training.merge import merge_and_save
            merge_and_save(model, tokenizer, config)
        trainer.accelerator.wait_for_everyone()

    # ==========================================================
    # STAGE 5: EVALUATION
    # ==========================================================
    if "evaluation" in stages:
        logger.info("=== Stage 5: Evaluation ===")

        # Load test set if not already available
        if test_df is None:
            test_path = os.path.join(
                config["training"].get("output_dir", "./checkpoints"),
                "test_locked.csv",
            )
            if not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"Locked test set not found at {test_path}. "
                    f"Run training first."
                )
            test_df = pd.read_csv(test_path)

        from pipeline.evaluation.evaluate.evaluator import (
            evaluate_all_directions,
        )

        eval_results = evaluate_all_directions(test_df, config)

        # Log to tracker
        from pipeline.benchmarking.tracker.wandb_tracker import WandBTracker
        if isinstance(tracker, WandBTracker):
            tracker.log_eval_results(
                [r.to_dict() for r in eval_results]
            )
        else:
            results_table = [
                [r.direction, r.n_samples, r.bleu, r.chrf_pp]
                for r in eval_results
            ]
            tracker.log_table(
                "evaluation/results",
                ["direction", "n_samples", "bleu", "chrf++"],
                results_table,
            )

        # Export results
        from utils.serializer.export_results import (
            export_to_json,
            export_to_csv,
        )
        results_dir = config.get("versioning", {}).get("output_dir", "./runs")
        export_to_json(
            eval_results,
            os.path.join(results_dir, "eval_results.json"),
            run_metadata=run_meta.to_dict() if run_meta else None,
        )
        export_to_csv(
            eval_results,
            os.path.join(results_dir, "eval_results.csv"),
        )

    # ==========================================================
    # BENCHMARKING (compare against previous runs)
    # ==========================================================
    if eval_results and run_meta:
        from pipeline.benchmarking.runner.benchmark_runner import (
            compare_against_previous,
        )
        comparison = compare_against_previous(
            eval_results, run_meta.to_dict(), config
        )
        if comparison:
            tracker.log_artifact(
                "benchmark_comparison",
                comparison,
                artifact_type="benchmarking",
            )

    # ==========================================================
    # SAVE FINAL VERSIONING METADATA
    # ==========================================================
    if run_meta and ver_cfg.get("enabled", True):
        from versioning.version_metadata import (
            update_with_results,
            save_run_metadata,
        )
        if eval_results:
            run_meta = update_with_results(
                run_meta, eval_results, training_duration
            )
        save_run_metadata(run_meta, config)

    # ── Finish ────────────────────────────────────────────────
    tracker.finish()
    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
