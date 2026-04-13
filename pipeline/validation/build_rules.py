"""
Build the list of validation rules from the YAML config.

This is the plug-and-play mechanism: add or remove rules in the config
without touching rule implementation code.
"""

from typing import Any, Dict, List

from pipeline.validation.rules import (
    Rule,
    min_length_rule,
    max_length_rule,
    length_ratio_rule,
    not_identical_rule,
    not_empty_or_numeric_rule,
    devanagari_ascii_ratio_rule,
)


def build_rules(config: Dict[str, Any]) -> List[Rule]:
    """
    Construct the ordered list of validation rules from config.

    Parameters
    ----------
    config : dict
        Full pipeline config. Reads ``config["validation"]``.

    Returns
    -------
    List[Rule]
        Each rule is ``(row) → bool``, returning True for valid rows.
    """
    val_cfg = config["validation"]
    rules: List[Rule] = []

    # Always apply length bounds
    rules.append(min_length_rule(val_cfg.get("min_word_len", 3)))
    rules.append(max_length_rule(val_cfg.get("max_word_len", 200)))
    rules.append(length_ratio_rule(val_cfg.get("max_length_ratio", 3.5)))

    # Identical source == target
    if val_cfg.get("reject_identical_pairs", True):
        rules.append(not_identical_rule())

    # Empty / numbers-only
    if val_cfg.get("reject_empty_or_numeric", True):
        rules.append(not_empty_or_numeric_rule())

    # Devanagari ASCII contamination check
    max_ascii = val_cfg.get("max_ascii_ratio_devanagari")
    if max_ascii is not None:
        rules.append(devanagari_ascii_ratio_rule(max_ascii))

    return rules
