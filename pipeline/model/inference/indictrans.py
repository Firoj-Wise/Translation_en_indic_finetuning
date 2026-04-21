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
        self.no_repeat_ngram_size = eval_cfg.get("no_repeat_ngram_size", 3)

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
        
        # IndicTrans2 expects inputs prefixed with "<src_lang> <tgt_lang> "
        formatted_texts = [f"{src_lang} {tgt_lang} {t}" for t in texts]
        
        results = []
        # Fallback to model's device
        device = getattr(self.model, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        for i in range(0, len(formatted_texts), batch_size):
            batch = formatted_texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            # Move inputs to device (handles nested dicts)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate!
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                )
            
            # Decode
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
            
        return results
