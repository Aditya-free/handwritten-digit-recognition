import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from streamlit_drawable_canvas import st_canvas


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
# 3. DESKEW HELPER
# ==================================================

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img
    skew = m['mu11'] / m['mu02']
    rows, cols = img.shape
    M = np.float32([[1, skew, -0.5 * cols * skew], [0, 1, 0]])
    deskewed = cv2.warpAffine(
        img, M, (cols, rows),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    return deskewed


# ==================================================
# 4. APPLICATION INTERFACE
# ==================================================

st.title("🔢 Handwritten Digit Recognition")

st.subheader("Deep Learning Project Using CNN")

st.write(
    "Upload a handwritten digit image from 0 to 9. "
    "The CNN model will predict the digit."
)

st.info(
    "For best results with notebook photos: ensure good lighting, "
    "hold the camera steady, and crop tightly around the digit. "
    "The app handles lined paper and uneven lighting automatically."
)


# ==================================================
# 5. UPLOAD IMAGE
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
    # 6. IMAGE CROPPING CONTROLS
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
        # 7. CROP IMAGE
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
        # 8. CONVERT TO GRAYSCALE
        # ==================================================

        gray = cv2.cvtColor(
            cropped_rgb,
            cv2.COLOR_RGB2GRAY
        )

        # ==================================================
        # 9. AUTO BRIGHTNESS / GAMMA CORRECTION
        # ==================================================

        mean_brightness = float(np.mean(gray))

        if mean_brightness < 80:
            gamma = 0.5  # brighten dark images
        elif mean_brightness > 180:
            gamma = 1.5  # darken washed-out images
        else:
            gamma = 1.0  # no correction needed

        if gamma != 1.0:
            gamma_corrected = (
                np.power(gray / 255.0, gamma) * 255.0
            ).astype(np.uint8)
        else:
            gamma_corrected = gray.copy()

        # ==================================================
        # 10. IMPROVE LOCAL CONTRAST (CLAHE)
        # ==================================================

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gamma_corrected)

        # ==================================================
        # 11. THRESHOLD THE IMAGE (ADAPTIVE + OTSU FALLBACK)
        # ==================================================

        # Dark handwriting becomes white foreground.
        # Light background becomes black.

        binary_adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10
        )

        total_pixels = binary_adaptive.shape[0] * binary_adaptive.shape[1]
        nonzero_adaptive = int(np.count_nonzero(binary_adaptive))

        if nonzero_adaptive < int(total_pixels * 0.01):
            # Foreground too sparse — fall back to Otsu
            _, binary_otsu = cv2.threshold(
                enhanced,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # Pick whichever gives more foreground pixels
            if int(np.count_nonzero(binary_otsu)) > nonzero_adaptive:
                binary = binary_otsu
            else:
                binary = binary_adaptive
        else:
            binary = binary_adaptive

        # ==================================================
        # 12. MORPHOLOGICAL CLOSING (FILL STROKE GAPS)
        # ==================================================

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        # ==================================================
        # 13. REMOVE LONG HORIZONTAL NOTEBOOK LINES
        # ==================================================

        count_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        line_mask_cc = np.zeros_like(binary)

        for i in range(1, count_cc):
            comp_w = stats_cc[i, cv2.CC_STAT_WIDTH]
            comp_h = stats_cc[i, cv2.CC_STAT_HEIGHT]
            # Only erase wide, thin components (notebook ruled lines)
            if comp_h > 0 and comp_w > 5 * comp_h:
                line_mask_cc[labels_cc == i] = 255

        # Expand the detected lines slightly then subtract
        line_mask_dilated = cv2.dilate(
            line_mask_cc,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )

        cleaned = cv2.subtract(binary, line_mask_dilated)

        # ==================================================
        # 14. REMOVE SMALL NOISE COMPONENTS
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
        # 15. LIGHT STROKE NORMALIZATION (SMOOTH JAGGED EDGES)
        # ==================================================

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )

        # ==================================================
        # 16. FIND THE DIGIT
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

        # ==================================================
        # 17. DESKEW
        # ==================================================

        digit = deskew(digit)

        # ==================================================
        # 18. ADD PADDING AROUND THE DIGIT
        # ==================================================

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
        # 19. RESIZE TO FIT A 20 x 20 AREA
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
        # 20. CENTER IN A 28 x 28 CANVAS
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
        # 21. QUALITY FEEDBACK
        # ==================================================

        nonzero_canvas = int(np.count_nonzero(canvas))

        if nonzero_canvas < 20:
            st.warning(
                "The processed image looks very sparse (fewer than 20 "
                "white pixels). The crop may be off, or the digit is "
                "very faint. Consider adjusting the crop sliders."
            )
        elif nonzero_canvas > 700:
            st.warning(
                "The processed image is very dense (more than 700 "
                "white pixels). There may be significant background "
                "noise. Consider adjusting the crop sliders."
            )

        # ==================================================
        # 22. PREPARE CNN INPUT
        # ==================================================

        img_array = canvas.astype(
            np.float32
        ) / 255.0

        img_array = img_array.reshape(
            1, 28, 28, 1
        )

        # ==================================================
        # 23. MAKE PREDICTION
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
        # 24. DISPLAY PROCESSED IMAGE
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
        # 25. DISPLAY PREDICTION
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
        # 26. DISPLAY ALL MODEL SCORES
        # ==================================================

        st.subheader("Scores for Digits 0–9")

        score_data = {
            str(i): float(predictions[i])
            for i in range(10)
        }

        st.bar_chart(score_data)

        # ==================================================
        # 27. SHOW IMAGE PROCESSING STEPS (DEBUG)
        # ==================================================

        with st.expander("Debug Image Processing"):

            st.image(
                cropped_rgb,
                caption="1. Cropped Original Image"
            )

            st.image(
                gamma_corrected,
                caption="2. Gamma Corrected"
            )

            st.image(
                enhanced,
                caption="3. Enhanced Grayscale (CLAHE)"
            )

            st.image(
                binary,
                caption="4. Thresholded (after closing)"
            )

            st.image(
                cleaned,
                caption="5. After Line Removal"
            )

            st.image(
                mask,
                caption="6. After Noise Removal"
            )

            st.image(
                digit,
                caption="7. Deskewed Digit (before 28×28 resize)"
            )

        st.caption(
            "The model score is not necessarily a calibrated "
            "probability. Incorrect predictions are possible."
        )

# ==================================================
# 28. DRAW DIGIT ON SCREEN
# ==================================================

st.divider()

st.header("✍️ Draw a Digit")

st.write(
    "Use your mouse to draw a digit from 0 to 9 "
    "in the white box below."
)

# Create a key for resetting the canvas
if "canvas_reset_counter" not in st.session_state:
    st.session_state.canvas_reset_counter = 0

# Clear button
if st.button("🧹 Clear Drawing"):
    st.session_state.canvas_reset_counter += 1
    st.rerun()

# Drawing canvas
canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    drawing_mode="freedraw",
    height=280,
    width=280,
    update_streamlit=True,
    key=f"digit_canvas_{st.session_state.canvas_reset_counter}",
    return_image_data=True
)

# Predict button
if st.button("🔍 Predict Drawn Digit"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")

    else:

        # Convert canvas image to an 8-bit array
        rgba = np.asarray(
            canvas_result.image_data
        ).astype(np.uint8)

        # Convert RGBA to grayscale
        gray = cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2GRAY
        )

        # Convert the black digit to white foreground
        # and the white background to black
        _, binary = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Find the digit boundaries
        points = cv2.findNonZero(binary)

        if points is None:
            st.warning("Please draw a digit first.")

        else:

            x, y, w, h = cv2.boundingRect(points)

            digit = binary[y:y + h, x:x + w]

            # Add padding around the digit
            padding = max(
                2,
                int(max(w, h) * 0.10)
            )

            digit = cv2.copyMakeBorder(
                digit,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=0
            )

            # Resize while preserving the aspect ratio
            digit_h, digit_w = digit.shape

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
                digit,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            # Center the digit in a 28 x 28 image
            processed = np.zeros(
                (28, 28),
                dtype=np.uint8
            )

            offset_x = (28 - new_w) // 2
            offset_y = (28 - new_h) // 2

            processed[
                offset_y:offset_y + new_h,
                offset_x:offset_x + new_w
            ] = resized

            # Normalize and prepare CNN input
            img_array = processed.astype(
                np.float32
            ) / 255.0

            img_array = img_array.reshape(
                1, 28, 28, 1
            )

            # Predict the digit
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_digit = int(
                np.argmax(predictions)
            )

            model_score = float(
                predictions[predicted_digit]
            ) * 100

            # Display processed image
            st.subheader("Processed Drawing")

            st.image(
                processed,
                caption="28 x 28 CNN Input",
                width=200
            )

            # Display result
            st.subheader("Prediction Result")

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.metric(
                "Model Score",
                f"{model_score:.2f}%"
            )

            # Display scores for all digits
            st.subheader("Scores for Digits 0–9")

            score_data = {
                str(i): float(predictions[i])
                for i in range(10)
            }

            st.bar_chart(score_data)
