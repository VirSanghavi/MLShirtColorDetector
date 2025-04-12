
Each folder contains 74+ images of that shirt color.

---

## 🧪 How It Works

1. Detect face using OpenCV Haar cascade
2. Select chest/shoulder region dynamically based on face
3. Resize and convert region to LAB color space
4. Extract average LAB color values
5. Classify using SVM model trained on labeled data
6. Display shirt color in live video feed

---

## 💻 Install & Run

### 1. Clone This Repo

### 2. Set Up Environment

python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

### 3. Train the ML Model

python3 train_model.py

### 4. Run the Live Detector

python3 shirt_color_detector.py

👨‍💻 Author
Vir Sanghavi
Founder of Up the Ratios, high school ML engineer, aspiring Stanford aerospace innovator.

📜 License
Use it, remix it, make it yours. Just give credit if you publish.

⭐ Star the Repo
If you learned something or found it helpful, star the repo to support the project!

### 5. Required Libraries
opencv-python
scikit-learn
joblib
numpy

