"""
IndicTrans2 inference wrapper.

Provides both single-text and batch translation using the
HuggingFace ``pipeline`` API with beam search parameters
from config.
"""

import logging
from typing import Any, Dict, List

import torch
from transformers import pipeline as hf_pipeline, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

logger = logging.getLogger("model.inference")


class IndicTransTranslator:
    """
    Reusable translation wrapper.

    Usage::

        translator = IndicTransTranslator(model, tokenizer, config)
        result = translator.translate("Hello, how are you?",
                                       src_lang="eng_Latn",
                                       tgt_lang="npi_Deva")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        eval_cfg = config.get("evaluation", {})
        self.num_beams = eval_cfg.get("num_beams", 5)
        self.max_length = eval_cfg.get("max_length", 256)
        self.length_penalty = eval_cfg.get("length_penalty", 1.0)
        self.no_repeat_ngram_size = eval_cfg.get("no_repeat_ngram_size", 0) # Defaults to 0 to prevent hallucination loops

        self.ip = IndicProcessor(inference=True)
        self.device = 0 if torch.cuda.is_available() else -1

    def translate(
        self,
        text: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "npi_Deva",
    ) -> str:
        """Translate a single text string."""
        return self.translate_batch([text], src_lang, tgt_lang, batch_size=1)[0]

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
    ) -> List[str]:
        """Translate a list of texts using direct generation to support QLoRA/PEFT cleanly."""
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        # Proper IndicTrans2 preprocessing (adds src_lang, tgt_lang tokens and standardizes punctuation)
        formatted_texts = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        
        results = []
        # Fallback to model's device
        device = getattr(self.model, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        for i in range(0, len(formatted_texts), batch_size):
            batch = formatted_texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            # Move inputs to device (handles nested dicts)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generation
            with torch.no_grad():
                gen_kwargs = {
                    "max_length": self.max_length,
                    "num_beams": self.num_beams,
                    "length_penalty": self.length_penalty,
                }
                
                # Only use no_repeat_ngram_size if it's explicitly enabled (> 0)
                # Setting this >0 often breaks Devanagari translation because proper grammar 
                # naturally repeats trigrams.
                if self.no_repeat_ngram_size > 0:
                    gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
                
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode
            decoded = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            # Proper IndicTrans2 postprocessing (handles devanagari punctuation mapping, un-tokenizes spaces)
            postprocessed = self.ip.postprocess_batch(decoded, lang=tgt_lang)
            results.extend(postprocessed)
            
        return results
