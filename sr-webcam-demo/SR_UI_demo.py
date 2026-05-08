import cv2
import time
import torch
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    half=(device == "cuda"),
    device=device,
)

WINDOW_NAME = "Real-ESRGAN Demo"
display_size = (640, 480)

show_menu = False
dev_view = False
show_metrics = False

cog_rect = None
dev_rect = None
metrics_rect = None

prev = time.time()


def draw_cog(frame):
    global cog_rect

    h, w = frame.shape[:2]
    size = 42
    x1 = w - size - 12
    y1 = 12
    x2 = x1 + size
    y2 = y1 + size

    cog_rect = (x1, y1, x2, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    cv2.putText(
        frame,
        "...",
        (x1 + 8, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )


def draw_menu(frame):
    global dev_rect, metrics_rect

    h, w = frame.shape[:2]

    menu_w = 210
    menu_h = 120
    x1 = w - menu_w - 12
    y1 = 62
    x2 = x1 + menu_w
    y2 = y1 + menu_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 35), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)

    dev_rect = (x1 + 10, y1 + 15, x2 - 10, y1 + 50)
    metrics_rect = (x1 + 10, y1 + 65, x2 - 10, y1 + 100)

    dev_label = "Dev view: ON" if dev_view else "Dev view: OFF"
    metrics_label = "Metrics: ON" if show_metrics else "Metrics: OFF"

    for rect, label in [(dev_rect, dev_label), (metrics_rect, metrics_label)]:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (70, 70, 70), -1)
        cv2.putText(
            frame,
            label,
            (rect[0] + 8, rect[1] + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )


def inside_rect(x, y, rect):
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def mouse_callback(event, x, y, flags, param):
    global show_menu, dev_view, show_metrics

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if inside_rect(x, y, cog_rect):
        show_menu = not show_menu
        return

    if show_menu and inside_rect(x, y, dev_rect):
        dev_view = not dev_view
        show_menu = False
        return

    if show_menu and inside_rect(x, y, metrics_rect):
        show_metrics = not show_metrics
        show_menu = False
        return


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lr = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        sr_frame, _ = upsampler.enhance(lr, outscale=2)

    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now

    if show_metrics:
        hr_resized = cv2.resize(frame, (sr_frame.shape[1], sr_frame.shape[0]))

        psnr_val = psnr(hr_resized, sr_frame)
        ssim_val = ssim(
            cv2.cvtColor(hr_resized, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(sr_frame, cv2.COLOR_BGR2GRAY),
        )

        sr_metrics = f"FPS: {fps:.1f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}"

    if dev_view:
        original_display = cv2.resize(frame, display_size)
        sr_display = cv2.resize(sr_frame, display_size)

        cv2.putText(original_display, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(sr_display, "SR", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if show_metrics:
            cv2.putText(sr_display, sr_metrics, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        output = cv2.hconcat([original_display, sr_display])

    else:
        output = cv2.resize(sr_frame, display_size)

        if show_metrics:
            cv2.putText(output, sr_metrics, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    draw_cog(output)

    if show_menu:
        draw_menu(output)

    cv2.imshow(WINDOW_NAME, output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()