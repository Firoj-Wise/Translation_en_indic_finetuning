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

    def _get_pipeline(self, src_lang: str, tgt_lang: str):
        """Build a HF translation pipeline for the given direction."""
        return hf_pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=self.device,
            max_length=self.max_length,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )

    def translate(
        self,
        text: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "npi_Deva",
    ) -> str:
        """Translate a single text string."""
        pipe = self._get_pipeline(src_lang, tgt_lang)
        return pipe(text)[0]["translation_text"]

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
    ) -> List[str]:
        """Translate a list of texts."""
        pipe = self._get_pipeline(src_lang, tgt_lang)
        results = pipe(texts, batch_size=batch_size)
        return [r["translation_text"] for r in results]
