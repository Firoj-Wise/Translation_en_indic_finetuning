"""
Load translation data from local parallel text files or pre-built CSV.

Refactored from ``scripts/build_csv.py`` — now config-driven.
"""

import logging
import os
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("ingestion.local")


def _build_from_text_files(data_dir: str) -> pd.DataFrame:
    """
    Read domain-split `.txt` files and assemble all 4 directions.

    Expected layout in ``data_dir``::

        Agriculture_english.txt
        Agriculture_nepali.txt
        Agriculture_maithili.txt
        Banking_english.txt
        ...

    Returns a DataFrame with columns:
        src, tgt, src_lang, tgt_lang, domain
    """
    rows: List[Dict[str, str]] = []

    # Discover domains by scanning for *_english.txt files
    domains = sorted({
        f.rsplit("_english.txt", 1)[0]
        for f in os.listdir(data_dir)
        if f.endswith("_english.txt")
    })

    if not domains:
        raise FileNotFoundError(
            f"No *_english.txt files found in {data_dir}. "
            f"Check your ingestion.local.data_dir config."
        )

    for domain in domains:
        en_path = os.path.join(data_dir, f"{domain}_english.txt")
        npi_path = os.path.join(data_dir, f"{domain}_nepali.txt")
        mai_path = os.path.join(data_dir, f"{domain}_maithili.txt")

        for path in (en_path, npi_path, mai_path):
            if not os.path.isfile(path):
                logger.warning(f"Missing file: {path} — skipping domain '{domain}'")
                break
        else:
            # All three files exist → read them
            with open(en_path, "r", encoding="utf-8") as f:
                en_lines = f.read().strip().split("\n")
            with open(npi_path, "r", encoding="utf-8") as f:
                npi_lines = f.read().strip().split("\n")
            with open(mai_path, "r", encoding="utf-8") as f:
                mai_lines = f.read().strip().split("\n")

            n = min(len(en_lines), len(npi_lines), len(mai_lines))
            logger.info(f"  {domain}: {n} aligned triplets")

            for i in range(n):
                en = en_lines[i].strip()
                npi = npi_lines[i].strip()
                mai = mai_lines[i].strip()

                if not en or not npi or not mai:
                    continue

                # eng → npi
                rows.append({"domain": domain, "src": en, "tgt": npi,
                             "src_lang": "eng_Latn", "tgt_lang": "npi_Deva"})
                # npi → eng
                rows.append({"domain": domain, "src": npi, "tgt": en,
                             "src_lang": "npi_Deva", "tgt_lang": "eng_Latn"})
                # eng → mai
                rows.append({"domain": domain, "src": en, "tgt": mai,
                             "src_lang": "eng_Latn", "tgt_lang": "mai_Deva"})
                # mai → eng
                rows.append({"domain": domain, "src": mai, "tgt": en,
                             "src_lang": "mai_Deva", "tgt_lang": "eng_Latn"})

    if not rows:
        raise ValueError("No valid sentence pairs assembled from text files.")

    return pd.DataFrame(rows)


def load_from_local(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data from local files. If ``csv_path`` is set, read that
    directly; otherwise assemble from parallel ``.txt`` files.

    Parameters
    ----------
    config : dict
        Full pipeline config. Reads ``config["ingestion"]["local"]``.

    Returns
    -------
    pd.DataFrame with columns: src, tgt, src_lang, tgt_lang, domain
    """
    local_cfg = config["ingestion"]["local"]
    csv_path = local_cfg.get("csv_path")

    if csv_path and os.path.isfile(csv_path):
        logger.info(f"Loading pre-built CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        data_dir = local_cfg["data_dir"]
        logger.info(f"Building dataset from text files in: {data_dir}")
        df = _build_from_text_files(data_dir)

    # Apply sample size limit if configured
    sample_size = config["ingestion"].get("sample_size")
    if sample_size and len(df) > sample_size:
        seed = config["pipeline"].get("seed", 42)
        df = df.sample(n=sample_size, random_state=seed)
        logger.info(f"Sampled {sample_size} rows for debug run")

    logger.info(f"Loaded {len(df)} rows from local source")
    return df.reset_index(drop=True)
