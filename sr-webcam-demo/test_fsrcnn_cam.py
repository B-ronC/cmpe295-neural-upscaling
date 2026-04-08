import cv2
from cv2 import dnn_superres
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------------
# Setup
# ---------------------------
MODEL_PATH = "model/FSRCNN_x2.pb"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Keep input small for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Load SR model
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("fsrcnn", 2)

# Display size (same for both)
display_size = (640, 480)

prev_time = time.time()

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------
    # Simulate LR input
    # ---------------------------
    hr = frame
    lr = cv2.resize(hr, (hr.shape[1] // 2, hr.shape[0] // 2))

    # ---------------------------
    # Super-resolution
    # ---------------------------
    sr_frame = sr.upsample(lr)

    # Resize HR to match SR size (for metrics)
    hr_resized = cv2.resize(hr, (sr_frame.shape[1], sr_frame.shape[0]))

    # ---------------------------
    # Metrics
    # ---------------------------
    psnr_val = psnr(hr_resized, sr_frame)
    ssim_val = ssim(
        cv2.cvtColor(hr_resized, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(sr_frame, cv2.COLOR_BGR2GRAY)
    )

    # ---------------------------
    # FPS
    # ---------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ---------------------------
    # Resize for display
    # ---------------------------
    input_display = cv2.resize(lr, display_size)
    sr_display = cv2.resize(sr_frame, display_size)

    # ---------------------------
    # Overlay text
    # ---------------------------
    text = f"FPS: {fps:.1f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}"

    cv2.putText(sr_display, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(input_display, "Input (LR)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(sr_display, "FSRCNN x2 Output", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---------------------------
    # Combine side-by-side
    # ---------------------------
    combined = cv2.hconcat([input_display, sr_display])

    cv2.imshow("Super Resolution Demo", combined)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()