import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pipeline.model.inference.indictrans import IndicTransTranslator

base_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
adapter_path = "Firoj112/indictrans2-en-npi-mai-finetuned"

print("1. Loading Base Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
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

inference_cfg = {
    "evaluation": {
        "num_beams": 5,
        "max_length": 256,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3
    }
}

base_translator = IndicTransTranslator(base_model, tokenizer, inference_cfg)

sentences = [
    "Micro-insurance provides protection to individuals with low income against specific perils.",
    "Farmers need better access to modern agricultural tools and fertilizers.",
    "The patient was admitted to the community hospital after experiencing a severe headache."
]

print("\n--- TRANSLATING WITH BASE MODEL ---")
base_np = base_translator.translate_batch(sentences, src_lang="eng_Latn", tgt_lang="npi_Deva", batch_size=4)
base_mai = base_translator.translate_batch(sentences, src_lang="eng_Latn", tgt_lang="mai_Deva", batch_size=4)

print("\n--- APPLYING LORA ADAPTER ---")
model = PeftModel.from_pretrained(base_model, adapter_path)
ft_translator = IndicTransTranslator(model, tokenizer, inference_cfg)

print("\n--- TRANSLATING WITH FINETUNED MODEL ---")
ft_np = ft_translator.translate_batch(sentences, src_lang="eng_Latn", tgt_lang="npi_Deva", batch_size=4)
ft_mai = ft_translator.translate_batch(sentences, src_lang="eng_Latn", tgt_lang="mai_Deva", batch_size=4)

print("\n================ COMPARISON ===================")
for i, src in enumerate(sentences):
    print(f"\n[EN]  {src}")
    print(f"[NP - BASE] {base_np[i]}")
    print(f"[NP - FINETUNED] {ft_np[i]}")
    print(f"[MAI - BASE] {base_mai[i]}")
    print(f"[MAI - FINETUNED] {ft_mai[i]}")
print("=================================================")
