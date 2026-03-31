import argparse
import os
import pandas as pd
import evaluate
import torch

from src.utils import set_seed, setup_logger, log_environment_info
from src.model import get_tokenizer, get_lora_model, merge_and_save_model
from src.data import clean_and_deduplicate, prepare_data
from src.train import get_training_args, setup_data_collator, RobustLoRATrainer
from src.eval import compute_metrics, full_evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="IndicTrans2 Finetuning Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw parallel corpus CSV (must have src, tgt, src_lang, tgt_lang, domain)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="./indictrans2-finetuned-final", help="Final merged model directory")
    parser.add_argument("--do_train", action="store_true", help="Run training pipeline")
    parser.add_argument("--do_eval", action="store_true", help="Run post-training evaluation across all directions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_run", action="store_true", help="Run with a small subset for debugging")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("IndicTrans2-Finetune")
    
    # 1. Environment & Reproducibility
    set_seed(args.seed)
    log_environment_info(logger)
    
    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)
        
        if args.sample_run:
            df = df.sample(min(1000, len(df)), random_state=args.seed)
            
        logger.info(f"Raw data size: {len(df)}")
        clean_df = clean_and_deduplicate(df)
        logger.info(f"Cleaned and deduped size: {len(clean_df)}")
        
        # Proper splits: 90/5/5
        clean_df = clean_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        train_end = int(0.9 * len(clean_df))
        val_end = int(0.95 * len(clean_df))
        
        train_df = clean_df.iloc[:train_end]
        val_df = clean_df.iloc[train_end:val_end]
        test_df = clean_df.iloc[val_end:]
        
        # Keep test_df safe
        test_df.to_csv(os.path.join(args.output_dir, "test_locked.csv"), index=False)
        logger.info(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # 3. Tokenizer and Model Integration
        tokenizer = get_tokenizer()
        
        train_dataset = prepare_data(train_df, tokenizer)
        val_dataset = prepare_data(val_df, tokenizer)
        
        model = get_lora_model()
        model.print_trainable_parameters()
        
        # Metrics setup
        bleu_metric = evaluate.load("sacrebleu")
        chrf_metric = evaluate.load("chrf")
        
        def compute_metrics_wrapper(eval_preds):
            return compute_metrics(eval_preds, tokenizer, bleu_metric, chrf_metric)
            
        data_collator = setup_data_collator(tokenizer)
        training_args = get_training_args(output_dir=args.output_dir, is_bf16=torch.cuda.is_bf16_supported())
        
        # 4. Trainer Initialization
        trainer = RobustLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_wrapper,
        )
        
        # 5. Training
        logger.info("Starting training...")
        trainer.train()
        
        # 6. Merge & Unload
        logger.info(f"Training complete. Merging LoRA and saving to {args.final_model_dir}")
        merge_and_save_model(model, tokenizer, args.final_model_dir)

    if args.do_eval:
        logger.info("Evaluating Final Model")
        test_df_path = os.path.join(args.output_dir, "test_locked.csv")
        if not os.path.exists(test_df_path):
            raise FileNotFoundError("Locked test set not found. Run training pipeline first to generate it.")
            
        test_df = pd.read_csv(test_df_path)
        
        # Re-verify on locked set
        scores = full_evaluate(
            model_dir=args.final_model_dir,
            tokenizer_dir=args.final_model_dir,
            test_df=test_df
        )
        for score in scores:
            logger.info(f"Final Evaluation Scores: {score}")

if __name__ == "__main__":
    main()
