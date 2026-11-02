# Handwritten Digit Recognition using CNN

A deep learning project that recognises handwritten digits (0–9) from uploaded photos or a live drawing canvas. Built with Python, TensorFlow/Keras, OpenCV, and Streamlit.

---

## Features

- 📷 **Upload a photo** of a handwritten digit — works with notebook paper, lined paper, and phone camera shots
- ✍️ **Draw a digit** directly in the browser using a canvas
- 🔢 **Predicts digits 0–9** using a trained Convolutional Neural Network (CNN)
- 📊 **Confidence bar chart** for all 10 digit classes
- 🔍 **Debug view** showing every preprocessing step (grayscale → threshold → line removal → 28×28 input)
- 📐 **Automatic preprocessing** for real-world photos:
  - Gamma correction for dark or overexposed images
  - CLAHE local contrast enhancement
  - Adaptive + Otsu thresholding (dual fallback)
  - Morphological closing to fill stroke gaps
  - Notebook ruled-line removal with aspect-ratio filtering
  - Deskewing to correct slight camera tilt
  - Connected-component noise removal

---

## Technologies

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| TensorFlow 2.x / Keras | CNN model training and inference |
| OpenCV | Image preprocessing pipeline |
| Streamlit | Web application UI |
| NumPy | Array operations |
| Pillow | Image loading |
| streamlit-drawable-canvas | In-browser drawing widget |

---

## Project Files

```
handwritten-digit-recognition/
├── app.py                        # Streamlit web application
├── train.py                      # CNN model training script
├── handwritten_digit_cnn.keras   # Saved trained model
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

The repository includes a pre-trained model (`handwritten_digit_cnn.keras`). To retrain from scratch with augmentation:

```bash
python train.py
```

Training uses the MNIST dataset (downloaded automatically). With augmentation and early stopping, expect ~20–25 epochs on a CPU or ~5 minutes on a GPU.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Architecture

The CNN is trained on MNIST with real-world augmentation to handle notebook photos:

```
Input (28×28×1)
│
├── Augmentation (training only)
│   ├── RandomRotation ±10%
│   ├── RandomZoom ±10%
│   └── RandomTranslation ±10%
│
├── Conv Block 1 — 32 filters × 2, BatchNorm, MaxPool, Dropout(0.25)
├── Conv Block 2 — 64 filters × 2, BatchNorm, MaxPool, Dropout(0.25)
├── Conv Block 3 — 128 filters × 1, BatchNorm, MaxPool, Dropout(0.25)
│
├── Flatten
├── Dense(256) + BatchNorm + Dropout(0.5)
└── Dense(10, softmax)
```

**Training augmentation also injects synthetic noise:**
- Gaussian pixel noise (simulates paper grain)
- Random brightness variation (simulates phone exposure)
- Random horizontal lines (simulates notebook ruled lines)

---

## Tips for Best Results with Photos

| ✅ Do | ❌ Avoid |
|---|---|
| Good lighting, no harsh shadows | Dark or very bright images |
| One digit per photo | Multiple digits in frame |
| Dark ink on light paper | Light ink or pencil on dark paper |
| Crop tightly around the digit | Large empty margins |
| Hold camera parallel to paper | Extreme angles |

The app handles ruled notebook lines automatically, but the closer you crop to the digit the better the result.

---

## How Preprocessing Works (Upload Path)

```
Original photo
    → Crop (user-defined sliders)
    → Grayscale conversion
    → Gamma correction (auto brightness)
    → CLAHE contrast enhancement
    → Adaptive threshold + Otsu fallback
    → Morphological closing (fill stroke gaps)
    → Horizontal line removal (aspect-ratio filtered)
    → Connected-component noise removal
    → Deskew (correct tilt)
    → Bounding box + padding
    → Resize to 20×20 preserving aspect ratio
    → Centre in 28×28 canvas
    → Normalize to [0, 1]
    → CNN prediction
```

---

## License

MIT
