import sys
import torch
from pipeline.model.inference.indictrans import IndicTransTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
adapter_path = "Firoj112/indictrans2-en-npi-mai-finetuned"

print("1. Loading Tokenizer...")
# Load tokenizer from the finetuned repo to avoid absolute path issues
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

print("2. Loading Base Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("3. Applying Finetuned Adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("4. Initializing Translator...")
inference_cfg = {
    "evaluation": {
        "num_beams": 5,
        "max_length": 256,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3
    }
}
translator = IndicTransTranslator(model, tokenizer, inference_cfg)

sentences = [
    "Micro-insurance provides protection to individuals with low income against specific perils.",
    "Farmers need better access to modern agricultural tools and fertilizers.",
    "The patient was admitted to the community hospital after experiencing a severe headache.",
    "You must link your bank account to the mobile application for quick transfers."
]

print("\n================ TRANSLATIONS ===================")
print("---- English to Nepali (npi_Deva) ----")
results_np = translator.translate_batch(sentences, src_lang="eng_Latn", tgt_lang="npi_Deva", batch_size=4)
for src, tgt in zip(sentences, results_np):
    print(f"EN: {src}")
    print(f"NP: {tgt}\n")

print("---- English to Maithili (mai_Deva) ----")
results_mai = translator.translate_batch(sentences, src_lang="eng_Latn", tgt_lang="mai_Deva", batch_size=4)
for src, tgt in zip(sentences, results_mai):
    print(f"EN: {src}")
    print(f"MAI: {tgt}\n")

print("=================================================")
