"""DSpark training dataset preparation."""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class DSparkDataset(Dataset):
    """Dataset for DSpark drafter training.

    Prepares (input_ids, attention_mask, loss_mask) triples from tokenized sequences.
    """

    def __init__(
        self,
        input_ids: list[list[int]],
        attention_mask: Optional[list[list[int]]] = None,
        max_length: int = 2048,
    ):
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        if attention_mask is not None:
            self.attention_mask = [torch.tensor(m, dtype=torch.long) for m in attention_mask]
        else:
            self.attention_mask = [
                torch.ones_like(ids) for ids in self.input_ids
            ]
        self.max_length = max_length

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx][: self.max_length],
            "attention_mask": self.attention_mask[idx][: self.max_length],
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Collate batch with padding."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Pad to max length in batch
    max_len = max(ids.shape[0] for ids in input_ids)

    padded_ids = torch.full(
        (len(batch), max_len),
        pad_token_id,
        dtype=torch.long,
    )
    padded_mask = torch.zeros(
        (len(batch), max_len),
        dtype=torch.long,
    )
    loss_mask = torch.zeros(
        (len(batch), max_len),
        dtype=torch.float32,
    )

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        seq_len = ids.shape[0]
        padded_ids[i, :seq_len] = ids
        padded_mask[i, :seq_len] = mask
        # Loss mask: only predict on non-padding tokens, but not the first one
        # (can't predict the first token without context)
        loss_mask[i, 1:seq_len] = 1.0

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "loss_mask": loss_mask,
    }


def prepare_dataset(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> DSparkDataset:
    """Prepare a DSparkDataset from raw texts.

    Args:
        texts: List of text strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        DSparkDataset with tokenized inputs.
    """
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return DSparkDataset(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized.get("attention_mask"),
        max_length=max_length,
    )


def get_dataloader(
    dataset: DSparkDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    pad_token_id: int = 0,
) -> DataLoader:
    """Get a DataLoader for DSpark training.

    Args:
        dataset: DSparkDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        pad_token_id: Padding token ID.

    Returns:
        DataLoader with proper collation.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
    )
