
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ==================================================
# 2. LOAD TRAINED CNN MODEL
# ==================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_cnn.keras"
    )


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the CNN model: {e}")
    st.stop()


# ==================================================
# 3. APPLICATION INTERFACE
# ==================================================

st.title("🔢 Handwritten Digit Recognition")

st.subheader("Deep Learning Project Using CNN")

st.write(
    "Upload a handwritten digit image from 0 to 9. "
    "The CNN model will predict the digit."
)

st.info(
    "For better results, upload a clear image with "
    "one dark handwritten digit on light paper."
)


# ==================================================
# 4. UPLOAD IMAGE
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    try:
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:
        st.error("Unable to open this image.")
        st.stop()

    st.image(
        original_image,
        caption="Original Uploaded Image",
        width=350
    )

    st.caption(
        "Tip: Crop your photo closely around the digit "
        "before uploading it. Avoid including large "
        "areas of the page."
    )

    # ==================================================
    # 5. IMAGE CROPPING CONTROLS
    # ==================================================

    st.subheader("Adjust Image Crop")

    st.write(
        "Use these controls to select the area containing "
        "your digit. The default values suit a digit near "
        "the center of a portrait photograph."
    )

    crop_top = st.slider(
        "Top (%)",
        0, 90, 15
    )

    crop_bottom = st.slider(
        "Bottom (%)",
        10, 100, 70
    )

    crop_left = st.slider(
        "Left (%)",
        0, 90, 10
    )

    crop_right = st.slider(
        "Right (%)",
        10, 100, 90
    )

    if crop_top >= crop_bottom or crop_left >= crop_right:
        st.warning("Please select a valid crop area.")
        st.stop()

    if st.button("Recognize Digit"):

        # ==================================================
        # 6. CROP IMAGE
        # ==================================================

        rgb = np.array(original_image)

        height, width = rgb.shape[:2]

        x1 = int(width * crop_left / 100)
        x2 = int(width * crop_right / 100)

        y1 = int(height * crop_top / 100)
        y2 = int(height * crop_bottom / 100)

        cropped_rgb = rgb[y1:y2, x1:x2]

        if cropped_rgb.size == 0:
            st.error("Invalid crop area.")
            st.stop()

        # ==================================================
        # 7. CONVERT TO GRAYSCALE
        # ==================================================

        gray = cv2.cvtColor(
            cropped_rgb,
            cv2.COLOR_RGB2GRAY
        )

        # Improve local contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # ==================================================
        # 8. THRESHOLD THE IMAGE
        # ==================================================

        # Dark handwriting becomes white foreground.
        # Light background becomes black.

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10
        )

        # ==================================================
        # 9. REMOVE LONG HORIZONTAL NOTEBOOK LINES
        # ==================================================

        kernel_width = max(
            40,
            int(binary.shape[1] * 0.45)
        )

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_width, 1)
        )

        horizontal_lines = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel
        )

        # Expand the detected lines slightly
        line_mask = cv2.dilate(
            horizontal_lines,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3)
            )
        )

        # Remove detected horizontal lines
        cleaned = cv2.subtract(
            binary,
            line_mask
        )

        # ==================================================
        # 10. REMOVE SMALL NOISE COMPONENTS
        # ==================================================

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                cleaned,
                connectivity=8
            )
        )

        mask = np.zeros_like(cleaned)

        for i in range(1, count):

            area = stats[i, cv2.CC_STAT_AREA]

            # Keep components large enough to be useful.
            # A low threshold helps preserve thin strokes.
            if area >= 5:
                mask[labels == i] = 255

        # ==================================================
        # 11. FIND THE DIGIT
        # ==================================================

        points = cv2.findNonZero(mask)

        if points is None:
            st.error(
                "No digit detected. Adjust the crop "
                "or upload a clearer image."
            )
            st.stop()

        x, y, w, h = cv2.boundingRect(points)

        digit = mask[y:y + h, x:x + w]

        # Add padding around the digit
        padding = max(
            2,
            int(max(w, h) * 0.10)
        )

        padded = cv2.copyMakeBorder(
            digit,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        # ==================================================
        # 12. RESIZE TO FIT A 20 x 20 AREA
        # ==================================================

        digit_h, digit_w = padded.shape

        scale = 20.0 / max(
            digit_h,
            digit_w
        )

        new_w = max(
            1,
            int(round(digit_w * scale))
        )

        new_h = max(
            1,
            int(round(digit_h * scale))
        )

        resized = cv2.resize(
            padded,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        # ==================================================
        # 13. CENTER IN A 28 x 28 CANVAS
        # ==================================================

        canvas = np.zeros(
            (28, 28),
            dtype=np.uint8
        )

        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2

        canvas[
            offset_y:offset_y + new_h,
            offset_x:offset_x + new_w
        ] = resized

        # ==================================================
        # 14. PREPARE CNN INPUT
        # ==================================================

        img_array = canvas.astype(
            np.float32
        ) / 255.0

        img_array = img_array.reshape(
            1, 28, 28, 1
        )

        # ==================================================
        # 15. MAKE PREDICTION
        # ==================================================

        try:
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        predicted_digit = int(
            np.argmax(predictions)
        )

        model_score = float(
            predictions[predicted_digit]
        ) * 100

        # ==================================================
        # 16. DISPLAY PROCESSED IMAGE
        # ==================================================

        st.subheader("Processed Image")

        st.write(
            "This is the 28 x 28 image passed to the CNN."
        )

        st.image(
            canvas,
            caption="CNN Input",
            width=280
        )

        # ==================================================
        # 17. DISPLAY PREDICTION
        # ==================================================

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.metric(
            "Model Score",
            f"{model_score:.2f}%"
        )

        # ==================================================
        # 18. DISPLAY ALL MODEL SCORES
        # ==================================================

        st.subheader("Scores for Digits 0–9")

        score_data = {
            str(i): float(predictions[i])
            for i in range(10)
        }

        st.bar_chart(score_data)

        # ==================================================
        # 19. SHOW IMAGE PROCESSING STEPS
        # ==================================================

        with st.expander("Debug Image Processing"):

            st.image(
                cropped_rgb,
                caption="1. Cropped Original Image"
            )

            st.image(
                enhanced,
                caption="2. Enhanced Grayscale Image"
            )

            st.image(
                binary,
                caption="3. Thresholded Image"
            )

            st.image(
                cleaned,
                caption="4. Image After Line Removal"
            )

            st.image(
                mask,
                caption="5. Image After Noise Removal"
            )

        st.caption(
            "The model score is not necessarily a calibrated "
            "probability. Incorrect predictions are possible."
        )
