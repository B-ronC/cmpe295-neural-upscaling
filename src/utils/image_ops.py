from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def list_image_files(directory: str) -> list[str]:
    path = Path(directory)
    return sorted(
        str(file)
        for file in path.iterdir()
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
