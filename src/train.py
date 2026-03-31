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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
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
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_steps=50,
        report_to="wandb",
        run_name="indictrans2-finetune",
        seed=42,
    )

def setup_data_collator(tokenizer):
    """Returns the data collator to handle padding properly."""
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
