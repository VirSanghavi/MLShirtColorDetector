import cv2
import numpy as np
from joblib import load
import time

# === Load model and label encoder ===
model = load("model_real/shirt_color_model.joblib")
label_encoder = load("model_real/label_encoder.joblib")

# === Load face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Start webcam ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("AI Shirt Color Detector", cv2.WINDOW_NORMAL)

print("⏳ Warming up camera...")
for _ in range(30):
    ret, frame = cap.read()
    if ret:
        break

shirt_color = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Chest/shoulder region
        sx = x - int(0.3 * w)
        ex = x + int(1.3 * w)
        sy = y + int(1.8 * h)
        ey = sy + int(0.6 * h)

        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(frame.shape[1], ex), min(frame.shape[0], ey)

        shirt_region = frame[sy:ey, sx:ex]
        if shirt_region.size > 0:
            resized = cv2.resize(shirt_region, (64, 64))
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            avg_lab = cv2.mean(lab)[:3]
            prediction = model.predict([avg_lab])
            shirt_color = label_encoder.inverse_transform(prediction)[0]
            print(f"🧠 ML Predicted: {shirt_color}")

            # Draw region + color patch
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            avg_bgr = cv2.mean(resized)[:3]
            patch = np.full((50, 50, 3), avg_bgr, dtype=np.uint8)
            frame[10:60, frame.shape[1]-60:frame.shape[1]-10] = patch
    else:
        cv2.putText(frame, "No face detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === Semi-transparent black banner ===
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    alpha = 0.4  # transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # === Bold black text over the banner ===
    cv2.putText(frame, f"Shirt Color: {shirt_color}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)


    cv2.imshow("AI Shirt Color Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("👋 Quitting...")
        break
    elif key == ord('r'):
        shirt_color = "Detecting..."

cap.release()
cv2.destroyAllWindows()
