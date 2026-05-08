# cmpe295-neural-upscaling
#  Webcam Super-Resolution Demo

A real-time webcam application that enhances video using **Real-ESRGAN** with optional UI controls and developer view.

---

## Features

- Real-time **super-resolution (x2)**
- Simple UI with toggle menu (top-right)
- Dev view (side-by-side original vs SR)
- Optional metrics (FPS, PSNR, SSIM)
- Runs on CPU or GPU (CUDA if available)

---

## Pipeline

Webcam → Real-ESRGAN → Display

---

##  Requirements

- Python **3.10**
- Windows (tested with DirectShow webcam)

---

##  Installation

### 1. Create virtual environment

```bash
py -3.10 -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip setuptools wheel

pip install numpy==1.26.4
pip install opencv-python==4.9.0.80

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install basicsr==1.4.2 realesrgan
pip install scikit-image==0.22.0 scipy==1.11.4 pillow==10.2.0

```
## Model Download

Download the pretrained Real-ESRGAN model:

https://github.com/xinntao/Real-ESRGAN/releases

After downloading, place it in:
weights/RealESRGAN_x2plus.pth

sr-webcam-demo/
├─ realesrgan_cam_demo.py
└─ weights/
└─ RealESRGAN_x2plus.pth
