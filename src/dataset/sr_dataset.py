import torch
import torchvision.transforms.functional as TF
from PIL import ImageOps
from torch.utils.data import Dataset

from utils.image_ops import ensure_min_size, list_image_files, load_rgb_image, make_lr_image, to_tensor


class SRTrainDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int = 4, crop_size: int = 64, crops_per_image: int = 5):
        self.hr_paths = list_image_files(hr_dir)
        self.scale = scale
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image

    def __len__(self) -> int:
        return len(self.hr_paths) * self.crops_per_image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        hr_image = load_rgb_image(self.hr_paths[index])
        lr_image = load_rgb_image(self.lr_paths[index])

        return {
            "pixel_values": to_tensor(lr_image),
            "labels": to_tensor(hr_image),
        }
