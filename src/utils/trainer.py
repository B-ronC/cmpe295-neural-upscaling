import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from transformers import Trainer

from utils.metrics import compute_sr_metrics


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
