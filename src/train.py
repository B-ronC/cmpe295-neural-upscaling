from pathlib import Path

from transformers import TrainingArguments

from dataset.collator import sr_collate_fn
from dataset.sr_dataset import SREvalDataset, SRTrainDataset
from models.simple_sr import SimpleSRModel
from utils.trainer import SRTrainer


def main():
    scale = 4
    data_root = Path("../data")
    output_dir = Path(f"../checkpoints/x{scale}")
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
