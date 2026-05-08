import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def y_channel(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def compute_sr_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    pred_y = y_channel(prediction)
    target_y = y_channel(target)

    psnr = peak_signal_noise_ratio(target_y, pred_y, data_range=1.0)
    ssim = structural_similarity(target_y, pred_y, data_range=1.0, channel_axis=None)
    return psnr, ssim
