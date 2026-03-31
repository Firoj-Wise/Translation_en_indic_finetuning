from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

def get_tokenizer(model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M") -> AutoTokenizer:
    """Load and return the IndicTrans2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return tokenizer

def get_lora_model(model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M", rank: int = 16) -> torch.nn.Module:
    """
    Loads the seq2seq model in full precision, applies Standard LoRA config.
    Uses target modules suitable for domain adaptation per guidelines.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    return model

def merge_and_save_model(model: torch.nn.Module, tokenizer: AutoTokenizer, output_dir: str):
    """
    Merges LoRA weights back into the base model and saves it along with the tokenizer.
    """
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
