import re
import unicodedata
import pandas as pd
from typing import Dict, Any, List
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def normalize_text(text: str) -> str:
    """
    Applies NFC normalization, removes zero-width chars, and collapses whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_pair(src: str, tgt: str, min_len: int = 3, max_len: int = 200, max_ratio: float = 3.5) -> bool:
    """
    Filters out bad pairs based on length, ratio, or identical strings.
    """
    s_tokens, t_tokens = src.split(), tgt.split()
    len_s, len_t = len(s_tokens), len(t_tokens)
    
    if len_s < min_len or len_t < min_len:
        return False
    if len_s > max_len or len_t > max_len:
        return False
        
    ratio = max(len_s, len_t) / max(min(len_s, len_t), 1)
    if ratio > max_ratio:
        return False
        
    if src.strip() == tgt.strip():
        return False
        
    return True

def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalization, filter invalid pairs, and deduplicate on the source side.
    Expects DataFrame with 'src' and 'tgt' columns.
    """
    df['src'] = df['src'].apply(normalize_text)
    df['tgt'] = df['tgt'].apply(normalize_text)
    
    valid_mask = df.apply(lambda row: is_valid_pair(row['src'], row['tgt']), axis=1)
    df = df[valid_mask].copy()
    
    df = df.drop_duplicates(subset=["src"])
    return df.reset_index(drop=True)

def preprocess_dataset_batch(batch: Dict[str, List[str]], tokenizer: AutoTokenizer, src_lang: str, tgt_lang: str) -> Dict[str, Any]:
    """
    Tokenizes the inputs and targets for a seq2seq dataset.
    Language tags must be appended properly by tokenizer.
    """
    inputs = tokenizer(
        batch["src"],
        src=True,
        max_length=256,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            batch["tgt"],
            src=False,
            max_length=256,
            truncation=True,
            padding=False,
        )
    inputs["labels"] = targets["input_ids"]
    return inputs

def prepare_data(df: pd.DataFrame, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Assumes df is already cleaned and split. 
    Handles multiple language directions by separating them, mapping, and combining.
    """
    from datasets import Dataset, concatenate_datasets
    from functools import partial
    
    all_datasets = []
    
    # Group by language directions to handle tokenization prefixes correctly
    for (src_lang, tgt_lang), group_df in df.groupby(['src_lang', 'tgt_lang']):
        group_dataset = Dataset.from_pandas(group_df)
        
        def tokenize_group(batch, src_lang=src_lang, tgt_lang=tgt_lang):
            # The IndicTrans2 custom tokenizer requires src_lang on the tokenizer itself
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            return preprocess_dataset_batch(batch, tokenizer, src_lang, tgt_lang)
            
        tokenized_group = group_dataset.map(
            tokenize_group,
            batched=True,
            num_proc=4,
            remove_columns=group_dataset.column_names
        )
        all_datasets.append(tokenized_group)
        
    return concatenate_datasets(all_datasets)
