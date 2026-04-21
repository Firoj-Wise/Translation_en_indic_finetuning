# IndicTrans2 — Translation Finetuning Pipeline

End-to-end pipeline for finetuning [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (`ai4bharat/indictrans2-en-indic-dist-200M`) on custom English ↔ Nepali / Maithili parallel corpora.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run the full pipeline (local data, default config)
uv run python run_pipeline.py --config configs/default.yaml

# 3. Quick debug run (1K subset, 1 epoch)
uv run python run_pipeline.py --config configs/sample_run.yaml
```

## Pipeline Stages

| Stage | Description |
|---|---|
| **Ingestion** | Load from HuggingFace Hub or local `.txt` / `.csv` files |
| **Validation** | Filter bad pairs, deduplicate, generate validation report |
| **Preprocessing** | NFC normalize, train/val/test split, tokenize per direction |
| **Training** | LoRA/QLoRA finetuning with W&B logging, cosine scheduler |
| **Evaluation** | Per-direction BLEU & chrF++ on locked test set |

## Configuration

All hyperparameters live in `configs/default.yaml`. Override any key from the CLI:

```bash
uv run python run_pipeline.py \
    --config configs/default.yaml \
    --override training.num_epochs=3 training.learning_rate=5e-5
```

### Key config sections

- `pipeline.stages` — which stages to run
- `ingestion.source` — `"local"` or `"huggingface"`
- `model.lora.*` — LoRA rank, alpha, target modules
- `training.*` — batch size, LR, scheduler, mixed precision
- `tracking.backend` — `"wandb"` or `"none"`

## Project Structure

```
├── configs/                     # YAML configurations
│   ├── default.yaml             # Master config
│   └── sample_run.yaml          # Debug config (1K subset)
├── constants/                   # Language codes, type definitions
├── pipeline/
│   ├── ingestion/               # HuggingFace + local data loaders
│   ├── validation/              # Plug-and-play pair filtering rules
│   ├── preprocessing/           # Normalization, tokenization, splitting
│   ├── model/
│   │   ├── loading/             # Tokenizer + LoRA/QLoRA model loader
│   │   └── inference/           # Translation pipeline wrapper
│   ├── training/                # Custom trainer, args, collator, callbacks
│   ├── evaluation/
│   │   ├── metrics/             # BLEU, chrF++ wrappers
│   │   └── evaluate/            # Trainer metric + post-training evaluator
│   └── benchmarking/
│       ├── tracker/             # W&B tracker abstraction
│       └── runner/              # Cross-run comparison
├── utils/                       # Seed, logging, serialization
├── versioning/                  # Content hashing, run metadata
├── run_pipeline.py              # Single entry point
└── pyproject.toml               # Dependencies
```

## Experiment Tracking

The pipeline logs everything to **Weights & Biases**:

- Training loss & eval metrics (BLEU, chrF++) per step
- GPU memory utilisation
- Per-direction evaluation table + bar charts
- Validation report as artifact
- Run metadata with config & dataset hashes

## Versioning

Each run produces a `run_metadata.json` in `./runs/<run_id>/` containing:

- UUID, git commit, timestamp
- SHA-256 hashes of config and dataset
- Final per-direction metrics
- Training duration

## Multi-GPU / Kaggle

```bash
# Multi-GPU with accelerate
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision fp16 \
    run_pipeline.py --config configs/default.yaml

# Push to Hub after training
uv run python run_pipeline.py --config configs/default.yaml \
    --override hub.push_to_hub=true hub.model_id=your-org/your-model
```

## Translation Directions

All 4 directions trained jointly in a single model:

| Direction | Source | Target |
|---|---|---|
| eng -> npi | `eng_Latn` | `npi_Deva` |
| npi -> eng | `npi_Deva` | `eng_Latn` |
| eng -> mai | `eng_Latn` | `mai_Deva` |
| mai -> eng | `mai_Deva` | `eng_Latn` |

## Inference / Interactive Mode

After finetuning, you can test translations in an interactive command-line interface. By default, it loads the base model in 4-bit (QLoRA) to save memory and applies your fine-tuned adapter.

```bash
uv run python inference/run_inference.py \
    --adapter_path Firoj112/indictrans2-en-npi-mai-finetuned \
    --quantize
```

**Features:**
- Type your sentences directly into the console.
- Specify source and target languages using `--src_lang` and `--tgt_lang` flags (defaults to `eng_Latn` -> `npi_Deva`).
- Type `quit` or `exit` to end the session.

To run a single translation without interaction:
```bash
uv run python inference/run_inference.py \
    --adapter_path Firoj112/indictrans2-en-npi-mai-finetuned \
    --quantize \
    --text "Agriculture is the backbone of our economy." \
    --src_lang eng_Latn \
    --tgt_lang npi_Deva
```

## Running as a Server (FastAPI)

If you want to host the model via a REST API:
```bash
# Start the Uvicorn server (defaults to port 8000)
uv run python serve.py

# Send requests (example via cURL)
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "How to apply for a loan in the bank?", "src_lang": "eng_Latn", "tgt_lang": "npi_Deva"}'
```

## End-to-End Workflow

### 1. Data Ingestion
By default, the pipeline looks for `.txt` files in `./finetuning_data/`. The files must be named `[Domain]_english.txt`, `[Domain]_nepali.txt`, and `[Domain]_maithili.txt`.

Alternatively, load directly from HuggingFace by overriding the data source:
```bash
uv run python run_pipeline.py --config configs/default.yaml \
    --override ingestion.source=huggingface \
               ingestion.huggingface.dataset_name=your-org/your-dataset
```

### 2. Training
Run the pipeline to execute validation, preprocessing, and training sequentially.

**Local (GPU Recommended):**
```bash
uv run python run_pipeline.py --config configs/default.yaml
```

**Kaggle / Multi-GPU:**
```bash
uv run accelerate launch --multi_gpu --num_processes=2 --mixed_precision fp16 \
    run_pipeline.py --config configs/default.yaml
```

### 3. Evaluation & Tracking
A locked test set (`test_locked.csv`) is preserved before training begins. After training, the `evaluation` stage computes corpus-level **BLEU** and **chrF++** scores for all 4 language directions on this test set.

**Viewing the Results:**
1. **Locally:** Results are logged in the console and saved to `./runs/<run_id>/eval_results.json` and `eval_results.csv`.
2. **Weights & Biases (Recommended):** The pipeline automatically logs everything to W&B. Open the run URL printed in your console to view:
   * **Training Curves:** Loss, learning rate, and eval metrics over time.
   * **GPU Metrics:** Memory utilization mapped to training steps.
   * **Evaluation Graphs:** Interactive bar charts showing BLEU and chrF++ scores per direction.
   * **Validation Artifacts:** Detailed JSON reports of rows dropped by each filter rule.
   * **Comparison:** Auto-generated tables comparing the current run's evaluation scores against previous bests.
