import cv2
from cv2 import dnn_superres
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

MODEL_PATH = "model/EDSR_x2.pb"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Keep it modest; EDSR is much heavier than FSRCNN
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("edsr", 2)

display_size = (640, 480)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hr = frame
    lr = cv2.resize(hr, (hr.shape[1] // 2, hr.shape[0] // 2))

    sr_frame = sr.upsample(lr)
    hr_resized = cv2.resize(hr, (sr_frame.shape[1], sr_frame.shape[0]))

    psnr_val = psnr(hr_resized, sr_frame)
    ssim_val = ssim(
        cv2.cvtColor(hr_resized, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(sr_frame, cv2.COLOR_BGR2GRAY)
    )

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    input_display = cv2.resize(lr, display_size)
    sr_display = cv2.resize(sr_frame, display_size)

    cv2.putText(input_display, "Input (LR)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(sr_display, "EDSR x2 Output", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(sr_display,
                f"FPS: {fps:.1f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    combined = cv2.hconcat([input_display, sr_display])
    cv2.imshow("EDSR Super Resolution Demo", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()