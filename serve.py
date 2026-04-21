from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pipeline.model.inference.indictrans import IndicTransTranslator
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Domain-Adapted IndicTrans2 API", description="English to Nepali & Maithili Translator")

# Global translator instance
translator = None

class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "eng_Latn"
    tgt_lang: str = "npi_Deva"  # Switch to 'mai_Deva' for Maithili

class TranslateResponse(BaseModel):
    original_text: str
    translation: str
    src_lang: str
    tgt_lang: str

@app.on_event("startup")
def load_translation_logic():
    global translator
    base_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    adapter_path = "Firoj112/indictrans2-en-npi-mai-finetuned"
    
    logger.info("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    
    logger.info("Loading Base Model efficiently using QLoRA (4-bit)...")
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
    
    logger.info("Injecting Domain-Adapted LoRA weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    inference_cfg = {
        "evaluation": {
            "num_beams": 5,
            "max_length": 256,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3
        }
    }
    translator = IndicTransTranslator(model, tokenizer, inference_cfg)
    logger.info("✅ Translation Server Ready!")

@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    if not translator:
        raise HTTPException(status_code=503, detail="Model is still loading. Please try again in a few seconds.")
    
    try:
        results = translator.translate_batch(
            [req.text], 
            src_lang=req.src_lang, 
            tgt_lang=req.tgt_lang, 
            batch_size=1
        )
        return TranslateResponse(
            original_text=req.text,
            translation=results[0],
            src_lang=req.src_lang,
            tgt_lang=req.tgt_lang
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation inference failed: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI server on port 8000...")
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
