import torch


def sr_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Received an empty batch.")

    label_shapes = [item["labels"].shape for item in batch]

    if len(set(label_shapes)) == 1:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    if len(batch) != 1:
        raise ValueError("Variable-size samples require batch_size=1.")

    sample = batch[0]
    return {
        "pixel_values": sample["pixel_values"].unsqueeze(0),
        "labels": sample["labels"].unsqueeze(0),
    }
