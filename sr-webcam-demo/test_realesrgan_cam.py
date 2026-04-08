import cv2
import time
import torch
import numpy as np
import contextlib
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

MODEL_PATH = "weights/RealESRGAN_x2plus.pth"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=2,
)

upsampler = RealESRGANer(
    scale=2,
    model_path=MODEL_PATH,
    model=model,
    tile=200,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device='cuda',
)

display_size = (640, 480) 
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hr = frame
    lr = cv2.resize(hr, (hr.shape[1] // 2, hr.shape[0] // 2))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        sr_frame, _ = upsampler.enhance(lr, outscale=2)

    hr_resized = cv2.resize(hr, (sr_frame.shape[1], sr_frame.shape[0]))

    psnr_val = psnr(hr_resized, sr_frame)
    ssim_val = ssim(
        cv2.cvtColor(hr_resized, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(sr_frame, cv2.COLOR_BGR2GRAY)
    )

    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now

    input_display = cv2.resize(lr, display_size)
    sr_display = cv2.resize(sr_frame, display_size)

    cv2.putText(input_display, "Input (LR)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(sr_display, "Real-ESRGAN x2", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(
        sr_display,
        f"FPS: {fps:.1f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    combined = cv2.hconcat([input_display, sr_display])
    cv2.imshow("Real-ESRGAN Demo", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()