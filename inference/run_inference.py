import argparse
import sys
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# Add project root to path so we can import from pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.model.inference.indictrans import IndicTransTranslator
from utils.logging.setup import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Run inference using fine-tuned IndicTrans2")
    parser.add_argument("--base_model", type=str, default="ai4bharat/indictrans2-en-indic-dist-200M", help="Path to base model or HF repo")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to directory containing PEFT adapter (e.g. checkpoints or HF repo tag)")
    parser.add_argument("--text", type=str, help="Text to translate. If not provided, starts interactive mode.")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source language (e.g. eng_Latn, npi_Deva)")
    parser.add_argument("--tgt_lang", type=str, default="npi_Deva", help="Target language (e.g. npi_Deva, eng_Latn)")
    parser.add_argument("--quantize", action="store_true", help="Load base model in 4-bit quantization (QLoRA) to save memory")
    
    args = parser.parse_args()

    logger = setup_logger("inference")
    logger.info(f"Loading tokenizer from adapter: {args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    
    logger.info(f"Loading base model: {args.base_model}")
    
    if args.quantize:
        logger.info("Using 4-bit quantization")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        logger.info("Using full precision")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model,
            trust_remote_code=True
        )
        
    logger.info(f"Applying PEFT adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    # Configuration based on default evaluation config
    config = {
        "evaluation": {
            "num_beams": 5,
            "max_length": 256,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3
        }
    }
    
    logger.info("Initializing translation pipeline Wrapper...")
    translator = IndicTransTranslator(model, tokenizer, config)
    logger.info("Model loaded successfully.")
    
    if args.text:
        result = translator.translate(args.text, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
        print(f"\nSource [{args.src_lang}]: {args.text}")
        print(f"Target [{args.tgt_lang}]: {result}\n")
    else:
        print("\n=== Interactive Translation Mode ===")
        print(f"Translating {args.src_lang} -> {args.tgt_lang}")
        print("Type 'quit' or 'exit' to stop.")
        while True:
            try:
                text = input(f"\nEnter text: ")
                if text.lower() in ("quit", "exit"):
                    break
                if not text.strip():
                    continue
                result = translator.translate(text, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
                print(f"Translation: {result}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during translation: {e}")

if __name__ == "__main__":
    main()
