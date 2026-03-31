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
    
    # Required for gradient checkpointing with PEFT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
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

def merge_and_save_model(model: torch.nn.Module, tokenizer: AutoTokenizer, output_dir: str, push_to_hub: bool = False, hub_model_id: str = None):
    """
    Merges LoRA weights back into the base model and saves it along with the tokenizer.
    Optionally pushes to Hugging Face Hub.
    """
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if push_to_hub and hub_model_id:
        print(f"Pushing merged model and tokenizer to Hugging Face Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
