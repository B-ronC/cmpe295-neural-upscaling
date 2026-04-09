import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def list_image_files(directory: str):
    path = Path(directory)
    return sorted(
        str(file) for file in path.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image).float()


def make_lr_image(hr_image: Image.Image, scale: int) -> Image.Image:
    width, height = hr_image.size
    lr_size = (width // scale, height // scale)
    return hr_image.resize(lr_size, Image.BICUBIC)


def ensure_min_size(image: Image.Image, min_size: int) -> Image.Image:
    width, height = image.size
    shortest_side = min(width, height)

    if shortest_side >= min_size:
        return image

    resize_ratio = min_size / shortest_side
    new_size = (int(width * resize_ratio), int(height * resize_ratio))
    return image.resize(new_size, Image.BICUBIC)


def y_channel(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def compute_sr_metrics(prediction: np.ndarray, target: np.ndarray):
    prediction = np.clip(prediction, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    pred_y = y_channel(prediction)
    target_y = y_channel(target)

    psnr = peak_signal_noise_ratio(target_y, pred_y, data_range=1.0)
    ssim = structural_similarity(target_y, pred_y, data_range=1.0, channel_axis=None)
    return psnr, ssim


class SRTrainDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int = 4, crop_size: int = 64):
        self.hr_paths = list_image_files(hr_dir)
        self.scale = scale
        self.crop_size = crop_size
        self.crops_per_image = 5

    def __len__(self):
        return len(self.hr_paths) * self.crops_per_image

    def __getitem__(self, index):
        image_index = index // self.crops_per_image
        crop_index = index % self.crops_per_image

        hr_image = load_rgb_image(self.hr_paths[image_index])
        hr_image = ensure_min_size(hr_image, self.crop_size)

        hr_crop = TF.five_crop(hr_image, self.crop_size)[crop_index]

        if torch.rand(1).item() < 0.5:
            hr_crop = ImageOps.mirror(hr_crop)

        rotation_k = torch.randint(0, 4, (1,)).item()
        if rotation_k:
            hr_crop = hr_crop.rotate(rotation_k * 90, expand=False)

        hr_crop = TF.center_crop(hr_crop, self.crop_size)
        lr_crop = make_lr_image(hr_crop, self.scale)

        return {
            "pixel_values": to_tensor(lr_crop),
            "labels": to_tensor(hr_crop),
        }


class SREvalDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4):
        self.hr_paths = list_image_files(hr_dir)
        self.lr_paths = list_image_files(lr_dir)
        self.scale = scale

        if len(self.hr_paths) != len(self.lr_paths):
            raise ValueError("HR and LR datasets must contain the same number of images.")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, index):
        hr_image = load_rgb_image(self.hr_paths[index])
        lr_image = load_rgb_image(self.lr_paths[index])

        return {
            "pixel_values": to_tensor(lr_image),
            "labels": to_tensor(hr_image),
        }


class SimpleSRModel(nn.Module):
    def __init__(self, scale: int = 4, num_features: int = 64, num_blocks: int = 8):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                )
                for _ in range(num_blocks)
            ]
        )
        self.tail = nn.Conv2d(64, 3 * scale * scale, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(scale)

    def forward(self, pixel_values, labels=None):
        _, _, lr_h, lr_w = pixel_values.shape
        target_size = (lr_h * self.scale, lr_w * self.scale)

        features = torch.relu(self.head(pixel_values))
        for block in self.body:
            features = features + block(features)

        sr = self.tail(features)
        sr = self.upsample(sr)

        base = F.interpolate(
            pixel_values,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        return sr + base


class SRTrainer(Trainer):
    @staticmethod
    def _match_output_size(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if outputs.shape[2:] == labels.shape[2:]:
            return outputs

        height, width = labels.shape[2:]
        return TF.center_crop(outputs, [height, width])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        outputs = self._match_output_size(outputs, labels)

        loss = F.l1_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        model = self.model
        device = self.args.device
        model.eval()

        losses = []
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values)
                outputs = self._match_output_size(outputs, labels)

                loss = F.l1_loss(outputs, labels)
                losses.append(loss.item())

                prediction = outputs.squeeze(0).cpu().numpy()
                target = labels.squeeze(0).cpu().numpy()

                psnr, ssim = compute_sr_metrics(prediction, target)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

        metrics = {
            f"{metric_key_prefix}_loss": float(np.mean(losses)),
            f"{metric_key_prefix}_psnr": float(np.mean(psnr_scores)),
            f"{metric_key_prefix}_ssim": float(np.mean(ssim_scores)),
        }

        self.log(metrics)
        return metrics


def sr_collate_fn(batch):
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


def main():
    scale = 4
    data_root = Path("./data")
    output_dir = Path(f"./checkpoints/x{scale}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SRTrainDataset(
        hr_dir=str(data_root / "train" / "BSDS200"),
        scale=scale,
        crop_size=64,
    )

    eval_dataset = SREvalDataset(
        hr_dir=str(data_root / "test" / "Set5" / "original"),
        lr_dir=str(data_root / "test" / "Set5" / "LRbicx4"),
        scale=scale,
    )

    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    model = SimpleSRModel(scale=scale)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        logging_steps=50,
        save_strategy="steps",
        save_steps=2000,
        eval_strategy="steps",
        eval_steps=2000,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = SRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=sr_collate_fn,
    )

    print("Starting training...")
    trainer.train()

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    print(f"Training finished. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()