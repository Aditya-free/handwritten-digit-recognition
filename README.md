# Handwritten Digit Recognition using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from uploaded images. It is built with Python, TensorFlow/Keras, OpenCV, and Streamlit.

## Features
- Upload an image containing a handwritten digit.
- Preprocess the image before prediction.
- Predict a digit from 0 to 9.
- Display prediction confidence scores.
- Visualize the prediction probabilities.

## Technologies Used
- Python
- TensorFlow and Keras
- OpenCV
- Streamlit
- NumPy

## Project Files
- `app.py` — Streamlit application.
- `handwritten_digit_cnn.keras` — trained CNN model.

## Run Locally

1. Clone this repository.
2. Install the required dependencies.
3. Run the application using Streamlit.

```bash
pip install streamlit tensorflow opencv-python numpy pillow
streamlit run app.py
```

## Current Limitations
Predictions may be inaccurate for photographs with shadows, notebook lines, complex backgrounds, or handwriting that differs from the training data.

## Future Improvements
- Improve automatic digit detection and cropping.
- Improve preprocessing for photographed handwriting.
- Evaluate performance on real-world handwritten samples.
- Explore training with a more diverse handwriting dataset.
