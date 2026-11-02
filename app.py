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
# 3. IMAGE PROCESSING HELPERS (FOR PHOTO UPLOAD)
# ==================================================
 
def extract_digit(rgb, thr=150):
    """
    Finds the handwritten digit in a photo automatically.
 
    Returns:
        digit_mask : binary mask (white digit on black), tightly cropped
        debug      : dict of intermediate images for the debug panel
    Returns (None, debug) if no digit is found.
    """
 
    debug = {}
 
    # --- Resize to a standard size so that all parameters
    #     behave the same for small and large photos
    h, w = rgb.shape[:2]
    s = 800.0 / max(h, w)
 
    rgb = cv2.resize(
        rgb,
        (max(1, int(w * s)), max(1, int(h * s))),
        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    )
 
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # --- Estimate the paper background (removes shadows and
    #     makes light blue notebook lines very faint).
    #     Done on a small image so it is fast and works even
    #     when the digit strokes are thick.
    small = cv2.resize(
        gray, None, fx=0.25, fy=0.25,
        interpolation=cv2.INTER_AREA
    )
 
    bg = cv2.morphologyEx(
        small,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
 
    bg = cv2.GaussianBlur(bg, (0, 0), 4)
 
    bg = cv2.resize(
        bg,
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
 
    # Paper becomes ~255, dark ink becomes low values
    norm = cv2.divide(gray, bg, scale=255)
    debug["1. Background normalised"] = norm
 
    # --- Threshold: only dark ink is kept
    binary = (norm < thr).astype(np.uint8) * 255
    debug["2. Ink only (threshold)"] = binary
 
    H, W = binary.shape
 
    # --- Join nearby strokes so one digit = one component
    joined = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
 
    count, labels, stats, cents = cv2.connectedComponentsWithStats(
        joined, connectivity=8
    )
 
    max_dist = np.hypot(W / 2, H / 2)
    best, best_score = None, 0.0
 
    for i in range(1, count):
 
        x, y, cw, ch, area = stats[i]
 
        # Too small -> noise
        if area < 0.0005 * H * W:
            continue
 
        # Long thin horizontal piece -> ruled line
        if cw > 4 * ch and ch < 0.08 * H:
            continue
 
        # Long thin vertical piece -> margin line
        if ch > 12 * cw and cw < 0.04 * W:
            continue
 
        # Huge blob touching the border -> shadow / table edge
        touches = x <= 1 or y <= 1 or x + cw >= W - 1 or y + ch >= H - 1
        if touches and area > 0.5 * H * W:
            continue
 
        # Prefer large components close to the image centre
        dist = np.hypot(cents[i][0] - W / 2, cents[i][1] - H / 2)
        score = area * (1.0 - 0.5 * dist / max_dist)
 
        if score > best_score:
            best, best_score = i, score
 
    if best is None:
        return None, debug
 
    digit = np.where(labels == best, binary, 0).astype(np.uint8)
 
    x, y, cw, ch = cv2.boundingRect(digit)
 
    if cw == 0 or ch == 0:
        return None, debug
 
    # Show the detected region on the original photo
    overlay = rgb.copy()
    cv2.rectangle(overlay, (x, y), (x + cw, y + ch), (255, 0, 0), 3)
    debug["3. Detected digit (red box)"] = overlay
 
    digit = digit[y:y + ch, x:x + cw]
 
    return digit, debug
 
 
def match_stroke_thickness(mask, side, ratio):
    """
    Makes the pen stroke about `ratio * side` pixels thick,
    similar to MNIST digits (and to the drawing canvas).
    """
 
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    vals = dist[dist > 0]
 
    if vals.size == 0:
        return mask
 
    stroke = 2.1 * np.percentile(vals, 95)
    target = ratio * side
    diff = int(round(target - stroke))
 
    if diff >= 2:
        k = diff + 1
        k += (k % 2 == 0)
        return cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
 
    if diff <= -3:
        k = abs(diff) + 1
        k += (k % 2 == 0)
        eroded = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        # Do not erode the digit away
        if np.count_nonzero(eroded) > 0.3 * np.count_nonzero(mask):
            return eroded
 
    return mask
 
 
def to_mnist(digit, ratio=0.13):
    """
    Converts a cropped white-on-black digit mask into a
    MNIST-style 28 x 28 image (20 x 20 digit, centred by
    centre of mass).
    """
 
    side = max(digit.shape)
    pad = int(side * 0.3) + 5
 
    d = cv2.copyMakeBorder(
        digit, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=0
    )
 
    d = match_stroke_thickness(d, side, ratio)
 
    x, y, w, h = cv2.boundingRect(d)
    d = d[y:y + h, x:x + w]
 
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
 
    d = cv2.resize(d, (new_w, new_h), interpolation=cv2.INTER_AREA)
 
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
 
    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2
    canvas28[oy:oy + new_h, ox:ox + new_w] = d
 
    # Centre by centre of mass (this is how MNIST was made)
    m = cv2.moments(canvas28)
 
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas28 = cv2.warpAffine(canvas28, M, (28, 28))
 
    return canvas28
 
 
# ==================================================
# 4. APPLICATION INTERFACE
# ==================================================
 
st.title("🔢 Handwritten Digit Recognition")
 
st.subheader("Deep Learning Project Using CNN")
 
st.write(
    "Upload a handwritten digit image from 0 to 9 "
    "or draw one below. The CNN model will predict the digit."
)
 
st.info(
    "For better results, write ONE digit with a dark pen, "
    "keep it reasonably large, and take the photo in good light."
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
        original_image = Image.open(uploaded_file).convert("RGB")
 
    except Exception:
        st.error("Unable to open this image.")
        st.stop()
 
    st.image(
        original_image,
        caption="Original Uploaded Image",
        width=350
    )
 
    # ------------------------------------------------
    # Settings
    # ------------------------------------------------
 
    thr = st.slider(
        "Ink sensitivity",
        80, 200, 150,
        help="Digit adhura aa raha ho to badhao. "
             "Lines / noise aa rahi ho to ghatao."
    )
 
    with st.expander("Manual crop (optional)"):
 
        st.write(
            "Digit apne aap detect hota hai. Sirf tab use karo "
            "jab photo mein aur bhi bahut kuch likha ho."
        )
 
        crop_top = st.slider("Top (%)", 0, 90, 0)
        crop_bottom = st.slider("Bottom (%)", 10, 100, 100)
        crop_left = st.slider("Left (%)", 0, 90, 0)
        crop_right = st.slider("Right (%)", 10, 100, 100)
 
    if crop_top >= crop_bottom or crop_left >= crop_right:
        st.warning("Please select a valid crop area.")
        st.stop()
 
    if st.button("Recognize Digit"):
 
        # ------------------------------------------------
        # Crop (default = full image)
        # ------------------------------------------------
 
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
 
        # ------------------------------------------------
        # Find the digit automatically
        # ------------------------------------------------
 
        digit, debug = extract_digit(cropped_rgb, thr)
 
        if digit is None:
            st.error(
                "No digit detected. Try a clearer photo, "
                "increase 'Ink sensitivity' or crop closer."
            )
 
            with st.expander("Debug Image Processing", expanded=True):
                for caption, img in debug.items():
                    st.image(img, caption=caption)
 
            st.stop()
 
        # ------------------------------------------------
        # Build 3 versions with different stroke thickness
        # and average the predictions (more stable result)
        # ------------------------------------------------
 
        ratios = [0.10, 0.13, 0.17]
 
        variants = [to_mnist(digit, r) for r in ratios]
 
        batch = np.stack(variants).astype(np.float32) / 255.0
        batch = batch.reshape(-1, 28, 28, 1)
 
        try:
            all_preds = model.predict(batch, verbose=0)
 
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
 
        predictions = all_preds.mean(axis=0)
 
        predicted_digit = int(np.argmax(predictions))
        model_score = float(predictions[predicted_digit]) * 100
 
        # ------------------------------------------------
        # Display
        # ------------------------------------------------
 
        st.subheader("Processed Image")
 
        st.write("This is the 28 x 28 image passed to the CNN.")
 
        st.image(variants[1], caption="CNN Input", width=280)
 
        st.subheader("Prediction Result")
 
        st.success(f"Predicted Digit: {predicted_digit}")
 
        st.metric("Model Score", f"{model_score:.2f}%")
 
        st.subheader("Scores for Digits 0–9")
 
        score_data = {
            str(i): float(predictions[i]) for i in range(10)
        }
 
        st.bar_chart(score_data)
 
        with st.expander("Debug Image Processing"):
 
            st.image(cropped_rgb, caption="0. Cropped Original Image")
 
            for caption, img in debug.items():
                st.image(img, caption=caption)
 
            st.write("CNN inputs with different stroke thickness:")
 
            cols = st.columns(3)
 
            for col, img, r in zip(cols, variants, ratios):
                col.image(img, caption=f"thickness {r}", width=130)
 
        st.caption(
            "The model score is not necessarily a calibrated "
            "probability. Incorrect predictions are possible."
        )
 
 
# ==================================================
# 6. DRAW DIGIT ON SCREEN
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
 
