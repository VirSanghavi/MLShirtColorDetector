import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def augment_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    brightness = np.random.randint(-30, 30)
    hsv[..., 2] = np.clip(hsv[..., 2] + brightness, 0, 255)
    hsv = hsv.astype(np.uint8)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    noise = np.random.normal(0, 15, bright.shape).astype(np.int16)
    noisy = np.clip(bright.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def extract_avg_lab(img):
    img = cv2.resize(img, (64, 64))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_lab = cv2.mean(lab)[:3]
    return np.array(avg_lab)

def load_dataset(folder_path):
    X, y = [], []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                for i in range(3):  # original + 2 augments
                    if i == 0:
                        features = extract_avg_lab(img)
                    else:
                        aug = augment_image(img)
                        features = extract_avg_lab(aug)
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

def train_model():
    X, y = load_dataset("real_shirts_dataset")
    print(f"✅ Loaded {len(X)} samples across {len(set(y))} labels.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = make_pipeline(SVC(kernel='rbf', probability=True))
    model.fit(X_train, y_train)

    os.makedirs("model_real", exist_ok=True)
    dump(model, "model_real/shirt_color_model.joblib")
    dump(le, "model_real/label_encoder.joblib")
    print("🎉 Trained ML model saved to 'model_real/'")

train_model()
