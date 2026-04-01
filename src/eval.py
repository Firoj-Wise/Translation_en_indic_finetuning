import evaluate
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def compute_metrics(eval_preds, tokenizer, bleu_metric, chrf_metric):
    """
    Standard HF Trainer evaluate callback. Note: Using sacrebleu and chrf.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
    
    return {
        "bleu": round(bleu_result["score"], 2),
        "chrf++": round(chrf_result["score"], 2)
    }

def full_evaluate(model_dir: str, tokenizer_dir: str, test_df: pd.DataFrame, batch_size: int = 32) -> list:
    """
    Post-training independent evaluator. Evaluates ALL directions found in test_df.
    Returns a list of metrics dictionaries per direction.
    """
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    device = 0 if torch.cuda.is_available() else -1
    
    results = []
    
    for (src_lang, tgt_lang), group_df in test_df.groupby(['src_lang', 'tgt_lang']):
        translator = pipeline(
            "translation", model=model, tokenizer=tokenizer,
            src_lang=src_lang, tgt_lang=tgt_lang,
            device=device, batch_size=batch_size,
            max_length=256, num_beams=5,
        )
        
        sources = group_df["src"].tolist()
        targets = [[t] for t in group_df["tgt"].tolist()]
        
        preds = [r["translation_text"] for r in translator(sources)]
        
        metrics = {
            "direction": f"{src_lang} → {tgt_lang}",
            "n": len(sources),
            "bleu": round(bleu_metric.compute(predictions=preds, references=targets)["score"], 2),
            "chrf++": round(chrf_metric.compute(predictions=preds, references=targets, word_order=2)["score"], 2),
        }
        results.append(metrics)
        
    return results

def generate_translation(text: str, model, tokenizer, src_lang: str = "eng_Latn", tgt_lang: str = "npi_Deva") -> str:
    """
    Inference helper with penalties as proposed.
    """
    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline(
        "translation",
        model=model, tokenizer=tokenizer,
        src_lang=src_lang, tgt_lang=tgt_lang,
        device=device,
        max_length=256,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
    )
    return translator(text)[0]["translation_text"]
