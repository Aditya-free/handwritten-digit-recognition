import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image, ImageOps
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
    return tf.keras.models.load_model("handwritten_digit_cnn.keras")


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the CNN model: {e}")
    st.stop()


# ==================================================
# 3. IMAGE PROCESSING HELPERS
# ==================================================

WORK_SIZE = 400  # photo crop is resized so its longest side is this many px


def photo_to_binary(rgb):
    """
    Notebook photo (RGB crop) -> binary image (white ink, black background).
    Returns (binary, debug_images).
    """
    debug = {}

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # --- 1. Resize to a fixed working size so kernel sizes are predictable
    h, w = gray.shape
    scale = WORK_SIZE / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=interp
    )

    # --- 2. Remove uneven lighting / paper shading
    #        (estimate the background, then divide it out)
    bg = cv2.dilate(
        gray,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    bg = cv2.medianBlur(bg, 31)
    bg = np.maximum(bg, 1)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    debug["1. Background-normalised gray"] = norm

    # --- 3. Otsu threshold: dark ink -> white
    _, binary = cv2.threshold(
        norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    debug["2. Thresholded"] = binary.copy()

    # --- 4. Remove thin horizontal notebook lines (even if they touch the digit)
    line_len = max(15, binary.shape[1] // 3)
    lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    )
    # keep thick parts (e.g. the bar of a 7 or 5), remove only thin parts
    thick = cv2.morphologyEx(
        lines,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    )
    thin_lines = cv2.subtract(lines, thick)
    thin_lines = cv2.dilate(
        thin_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    )
    binary = cv2.subtract(binary, thin_lines)

    # --- 5. Close small gaps left in the strokes
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    debug["3. After line removal + closing"] = binary.copy()

    # --- 6. Keep only the main digit (drop stray noise)
    binary = keep_main_digit(binary)
    debug["4. Main digit only"] = binary.copy()

    return binary, debug


def keep_main_digit(binary):
    """Keep the largest component plus big pieces close to it."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if n <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    lx, ly, lw, lh = stats[largest, :4]
    max_area = areas.max()

    keep = np.zeros_like(binary)

    for i in range(1, n):
        if i == largest:
            keep[labels == i] = 255
            continue

        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]

        near = (
            lx - 0.5 * lw <= cx <= lx + 1.5 * lw
            and ly - 0.5 * lh <= cy <= ly + 1.5 * lh
        )

        # broken strokes (like in 4, 5, 7) are kept; tiny specks are dropped
        if near and area >= 0.10 * max_area:
            keep[labels == i] = 255

    return keep


def thicken_strokes(digit, size, target_ratio=0.08):
    """
    Pen strokes are thin compared to MNIST / canvas strokes.
    Estimate the stroke width and dilate until it is ~8% of digit size.
    """
    dist = cv2.distanceTransform(digit, cv2.DIST_L2, 3)
    vals = dist[digit > 0]

    if vals.size == 0:
        return digit

    stroke = 4.0 * float(np.mean(vals))  # approx. stroke width in px
    target = target_ratio * size

    if stroke >= target:
        return digit

    add = int(round((target - stroke) / 2))
    add = min(add, max(1, int(size * 0.06)))

    if add < 1:
        return digit

    return cv2.dilate(
        digit,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * add + 1, 2 * add + 1)
        )
    )


def to_28x28(binary, thicken=False):
    """
    White-on-black binary image -> 28x28 MNIST-style image.
    Used by BOTH the photo path and the canvas path.
    """
    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    digit = binary[y:y + h, x:x + w]

    size = max(w, h)
    padding = max(2, int(size * 0.10))

    digit = cv2.copyMakeBorder(
        digit, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )

    if thicken:
        digit = thicken_strokes(digit, size)

    digit_h, digit_w = digit.shape
    scale = 20.0 / max(digit_h, digit_w)

    new_w = max(1, int(round(digit_w * scale)))
    new_h = max(1, int(round(digit_h * scale)))

    resized = cv2.resize(
        digit, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    out = np.zeros((28, 28), dtype=np.uint8)
    off_x = (28 - new_w) // 2
    off_y = (28 - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    return out


def predict_digit(img28):
    arr = img28.astype(np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return model.predict(arr, verbose=0)[0]


def show_prediction(predictions):
    predicted_digit = int(np.argmax(predictions))
    score = float(predictions[predicted_digit]) * 100

    st.subheader("Prediction Result")
    st.success(f"Predicted Digit: {predicted_digit}")
    st.metric("Model Score", f"{score:.2f}%")

    st.subheader("Scores for Digits 0–9")
    st.bar_chart({str(i): float(predictions[i]) for i in range(10)})


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
    "hold the camera steady, write with a dark pen, and crop tightly "
    "around the digit."
)


# ==================================================
# 5. UPLOAD IMAGE
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    original_image = None

    try:
        original_image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")
    except Exception:
        st.error("Unable to open this image.")

    if original_image is not None:

        st.image(
            original_image,
            caption="Original Uploaded Image",
            width=350
        )

        # ----------------------------------------------
        # 6. CROP CONTROLS
        # ----------------------------------------------

        st.subheader("Adjust Image Crop")
        st.write(
            "Select the area containing your digit. "
            "Keep it as tight as possible."
        )

        crop_top = st.slider("Top (%)", 0, 90, 15)
        crop_bottom = st.slider("Bottom (%)", 10, 100, 70)
        crop_left = st.slider("Left (%)", 0, 90, 10)
        crop_right = st.slider("Right (%)", 10, 100, 90)

        crop_valid = (
            crop_top < crop_bottom and crop_left < crop_right
        )

        if not crop_valid:
            st.warning("Please select a valid crop area.")

        if crop_valid and st.button("Recognize Digit"):

            rgb = np.array(original_image)
            height, width = rgb.shape[:2]

            x1 = int(width * crop_left / 100)
            x2 = int(width * crop_right / 100)
            y1 = int(height * crop_top / 100)
            y2 = int(height * crop_bottom / 100)

            cropped_rgb = rgb[y1:y2, x1:x2]

            if cropped_rgb.size == 0:
                st.error("Invalid crop area.")
            else:
                binary, debug = photo_to_binary(cropped_rgb)
                canvas28 = to_28x28(binary, thicken=True)

                if canvas28 is None:
                    st.error(
                        "No digit detected. Adjust the crop "
                        "or upload a clearer image."
                    )
                else:
                    nz = int(np.count_nonzero(canvas28))

                    if nz < 20:
                        st.warning(
                            "The processed image is very sparse. "
                            "The crop may be off, or the digit is faint."
                        )
                    elif nz > 700:
                        st.warning(
                            "The processed image is very dense. "
                            "There may be background noise."
                        )

                    predictions = predict_digit(canvas28)

                    st.subheader("Processed Image")
                    st.write("This is the 28 x 28 image passed to the CNN.")
                    st.image(canvas28, caption="CNN Input", width=280)

                    show_prediction(predictions)

                    with st.expander("Debug Image Processing"):
                        st.image(cropped_rgb, caption="0. Cropped Original")
                        for caption, img in debug.items():
                            st.image(img, caption=caption)
                        st.image(canvas28, caption="5. Final 28×28 input")

                    st.caption(
                        "The model score is not necessarily a calibrated "
                        "probability. Incorrect predictions are possible."
                    )


# ==================================================
# 7. DRAW DIGIT ON SCREEN
# ==================================================

st.divider()
st.header("✍️ Draw a Digit")

st.write(
    "Use your mouse to draw a digit from 0 to 9 "
    "in the white box below."
)

if "canvas_reset_counter" not in st.session_state:
    st.session_state.canvas_reset_counter = 0

if st.button("🧹 Clear Drawing"):
    st.session_state.canvas_reset_counter += 1
    st.rerun()

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

if st.button("🔍 Predict Drawn Digit"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        rgba = np.asarray(canvas_result.image_data).astype(np.uint8)
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        processed = to_28x28(binary, thicken=False)

        if processed is None:
            st.warning("Please draw a digit first.")
        else:
            predictions = predict_digit(processed)

            st.subheader("Processed Drawing")
            st.image(processed, caption="28 x 28 CNN Input", width=200)

            show_prediction(predictions)
