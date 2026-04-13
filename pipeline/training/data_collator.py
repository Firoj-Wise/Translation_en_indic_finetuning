"""
Custom DataCollator for IndicTrans2.

The IndicTrans2 model lacks ``prepare_decoder_input_ids_from_labels``,
so we manually construct ``decoder_input_ids`` by shifting labels right
and prepending the decoder start token.
"""

import torch
from transformers import DataCollatorForSeq2Seq


class IndicTransDataCollator(DataCollatorForSeq2Seq):
    """
    Extends ``DataCollatorForSeq2Seq`` to manually create
    ``decoder_input_ids`` when the model doesn't provide them.
    """

    def __call__(self, features, return_tensors=None):
        features = super().__call__(features, return_tensors=return_tensors)

        if "labels" in features and "decoder_input_ids" not in features:
            labels = features["labels"]
            # IndicTrans2 uses eos_token_id (2) as decoder start token
            decoder_start_token_id = self.tokenizer.eos_token_id

            decoder_input_ids = labels.clone()
            # Replace padding marker (-100) with pad_token_id
            decoder_input_ids[decoder_input_ids == -100] = (
                self.tokenizer.pad_token_id
            )

            # Shift right: prepend decoder_start_token_id
            shifted = torch.zeros_like(decoder_input_ids)
            shifted[:, 1:] = decoder_input_ids[:, :-1]
            shifted[:, 0] = decoder_start_token_id

            features["decoder_input_ids"] = shifted

        return features


def build_data_collator(tokenizer) -> IndicTransDataCollator:
    """
    Factory for the IndicTrans2 data collator.

    - ``padding=True`` → dynamic padding per batch (efficient)
    - ``pad_to_multiple_of=8`` → align to tensor-core boundaries
    - ``label_pad_token_id=-100`` → ignored by CrossEntropyLoss
    """
    return IndicTransDataCollator(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
