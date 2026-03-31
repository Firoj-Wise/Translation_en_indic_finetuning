import math
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

def get_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.05):
    """Cosine annealing with linear warmup and minimum learning rate floor."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

class RobustLoRATrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that implements the required warmup and cosine scheduler.
    (Note: Discriminative LR and Gradual Unfreezing are omitted here because
    we are strictly following the LoRA methodology per the user prompt.
    When using LoRA, the adapter matrices are the only things that train.)
    """
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        self.lr_scheduler = get_warmup_cosine_scheduler(
            self.optimizer, 
            num_warmup_steps, 
            num_training_steps, 
            min_lr_ratio=0.05
        )

def get_training_args(output_dir: str = "./checkpoints", is_fp16: bool = False, is_bf16: bool = True) -> Seq2SeqTrainingArguments:
    """Deliberate hyperparameters for IndicTrans2 LoRA finetuning."""
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        label_smoothing_factor=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=is_bf16,
        fp16=is_fp16,
        gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf++",
        greater_is_better=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        logging_steps=50,
        report_to="wandb",
        run_name="indictrans2-finetune",
        seed=42,
    )

class IndicTransDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features = super().__call__(features, return_tensors=return_tensors)
        
        # Manually create decoder_input_ids from labels if they don't exist
        # because IndicTrans2 model lacks prepare_decoder_input_ids_from_labels
        if "labels" in features and "decoder_input_ids" not in features:
            labels = features["labels"]
            decoder_start_token_id = self.tokenizer.eos_token_id  # which is 2

            decoder_input_ids = labels.clone()
            # replace padding (-100) with pad_token_id
            decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id
            
            # shift right
            decoder_input_ids_shifted = torch.zeros_like(decoder_input_ids)
            decoder_input_ids_shifted[:, 1:] = decoder_input_ids[:, :-1]
            decoder_input_ids_shifted[:, 0] = decoder_start_token_id
            
            features["decoder_input_ids"] = decoder_input_ids_shifted
            
        return features

def setup_data_collator(tokenizer):
    """Returns the data collator to handle padding properly."""
    return IndicTransDataCollator(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
