import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image

# Register any component manifests for Streamlit 1.66+ components
try:
    from streamlit.components.v2 import get_bidi_component_manager
    from streamlit.components.v2.manifest_scanner import scan_component_manifests
    _mgr = get_bidi_component_manager()
    for _m, _pkg_root in scan_component_manifests():
        _mgr.register_from_manifest(_m, _pkg_root)
except Exception:
    pass

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


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
# 3. ADVANCED IMAGE PREPROCESSING PIPELINE
# ==================================================

def center_by_center_of_mass(canvas_28x28):
    """
    Shifts the 28x28 digit canvas so its center of mass is at (14, 14),
    matching the official MNIST dataset standard.
    """
    M = cv2.moments(canvas_28x28)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        shift_x = max(-4, min(4, shift_x))
        shift_y = max(-4, min(4, shift_y))
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(
            canvas_28x28,
            M_shift,
            (28, 28),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    return canvas_28x28


def process_digit_patch(digit_mask):
    """
    Takes an isolated binary mask (white digit on black background),
    normalizes stroke thickness to match MNIST, scales to 20x20,
    and centers it into a 28x28 canvas using Center of Mass.
    """
    pts = cv2.findNonZero(digit_mask)
    if pts is None:
        return None

    dx, dy, dw, dh = cv2.boundingRect(pts)
    digit_crop = digit_mask[dy:dy + dh, dx:dx + dw]

    # Normalize stroke thickness for camera photos
    # MNIST strokes in 28x28 are ~2.5-3.5 px (~12-15% of digit height)
    target_stroke = max(1, int(round(dh * 0.08)))
    if target_stroke > 3:
        dilate_size = max(3, target_stroke // 2) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        digit_crop = cv2.dilate(digit_crop, kernel)

    # Pad around the digit
    pad = max(4, int(max(dw, dh) * 0.15))
    padded = cv2.copyMakeBorder(
        digit_crop, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=0
    )

    # Scale to fit inside 20x20 preserving aspect ratio
    scale = 20.0 / max(padded.shape)
    new_w = max(1, int(round(padded.shape[1] * scale)))
    new_h = max(1, int(round(padded.shape[0] * scale)))
    resized = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox = (28 - new_w) // 2
    oy = (28 - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    # Center of mass alignment
    return center_by_center_of_mass(canvas)


def extract_digit_from_photo(img_rgb):
    """
    Robustly segments a handwritten digit from plain paper or lined notebook photos:
    1. Illumination normalization (removes shadows, paper grain, uneven lighting)
    2. Dynamic thresholding (preserves faint pen ink)
    3. Notebook ruled-line removal via HoughLinesP and morphology
    4. Stroke bridging (heals cuts through digits where ruled lines passed)
    5. Connected-component candidate filtering & nearby stroke merging
    6. Center-of-mass aligned 28x28 MNIST canvas generation
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. Illumination estimation using Gaussian Blur
    ksize = max(25, int(min(h, w) * 0.08) | 1)
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Ink difference: ink is darker than paper, so (bg - gray) > 0 for ink
    if np.mean(gray) > 120:
        diff = cv2.subtract(bg, gray)
    else:
        diff = cv2.subtract(gray, bg)

    # 2. Dynamic thresholding based on Otsu
    otsu_val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(15, int(otsu_val * 0.70))
    _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Remove horizontal notebook ruled lines using HoughLinesP & Morphology
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180,
        threshold=70,
        minLineLength=int(w * 0.25),
        maxLineGap=30
    )
    line_mask = np.zeros_like(binary)
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = l
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Ruled notebook lines are nearly horizontal (within +/- 12 degrees)
            if abs(angle) < 12 or abs(angle - 180) < 12 or abs(angle + 180) < 12:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 4)

    # Secondary morphological horizontal line detector
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.20), 1))
    morph_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    line_mask = cv2.bitwise_or(line_mask, morph_lines)

    # Subtract detected lines from binary image
    binary_no_lines = cv2.subtract(binary, line_mask)

    # 4. Stroke bridging: restore strokes that were cut by notebook lines
    vert_bridge = cv2.morphologyEx(
        binary_no_lines,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    )
    stroke_bridge = cv2.morphologyEx(
        vert_bridge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    # 5. Connected components on cleaned image
    num_comps, labels, stats, _ = cv2.connectedComponentsWithStats(stroke_bridge)
    if num_comps <= 1:
        return None, None, None, "No handwritten digit found in the image."

    # Filter candidate components: eliminate edge noise, table borders, or ruled line fragments
    candidates = []
    for i in range(1, num_comps):
        cx, cy, cw, ch, ca = stats[i]
        # Digits are not the full width of the image
        if cw > w * 0.75:
            continue
        # Digits are not located completely on the outer border
        if cy + ch > h * 0.95 or cy < h * 0.02:
            if ch < 100:
                continue
        if ca >= 40:
            aspect = cw / max(1, ch)
            if 0.15 < aspect < 3.0:
                candidates.append((i, ca, (cx, cy, cw, ch)))

    if not candidates:
        return None, None, None, "Could not isolate a valid digit candidate."

    # Pick the primary digit component (largest area among valid candidates)
    best_candidate = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    best_idx = best_candidate[0]
    mx, my, mw, mh = best_candidate[2]

    digit_mask = (labels == best_idx).astype(np.uint8) * 255

    # Merge nearby components belonging to the same digit (e.g. crossbars of 4, 5, 7)
    margin = int(max(mw, mh) * 0.35)
    for other_idx, oa, (ox, oy, ow, oh) in candidates:
        if other_idx == best_idx:
            continue
        if (ox + ow >= mx - margin and ox <= mx + mw + margin and
            oy + oh >= my - margin and oy <= my + mh + margin):
            digit_mask[labels == other_idx] = 255

    pts = cv2.findNonZero(digit_mask)
    if pts is None:
        return None, None, None, "Could not locate digit boundaries."

    bx, by, bw, bh = cv2.boundingRect(pts)
    bbox = (bx, by, bw, bh)

    # Create 28x28 canvas
    canvas = process_digit_patch(digit_mask)
    if canvas is None:
        return None, None, None, "Failed to build 28x28 representation."

    debug_info = {
        "diff": diff,
        "line_mask": line_mask,
        "stroke_bridge": stroke_bridge,
        "digit_mask": digit_mask,
        "bbox": bbox
    }

    return canvas, bbox, debug_info, None


# ==================================================
# 4. APPLICATION INTERFACE
# ==================================================

st.title("🔢 Handwritten Digit Recognition")
st.subheader("Deep Learning Project Using CNN")

st.write(
    "Upload a photo of a handwritten digit from plain paper or lined notebook, "
    "or draw one directly on the canvas below."
)

st.info(
    "💡 **Automatic Detection & Ruled-Line Removal**: The app automatically detects "
    "the digit, erases notebook lines, heals cut strokes, and removes shadows."
)


# ==================================================
# 5. UPLOAD IMAGE SECTION
# ==================================================

uploaded_file = st.file_uploader(
    "Choose a digit image (photo of notebook or paper)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Unable to open this image file.")
        st.stop()

    rgb_full = np.array(original_image)
    h_full, w_full = rgb_full.shape[:2]

    st.image(
        original_image,
        caption=f"Uploaded Image ({w_full} × {h_full} px)",
        use_container_width=True
    )

    # Optional manual crop
    with st.expander("✂️ Optional: Adjust Crop Area (only needed for multi-digit pages)"):
        use_manual_crop = st.checkbox("Enable manual cropping", value=False)
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            crop_top = st.slider("Top (%)", 0, 90, 0)
            crop_left = st.slider("Left (%)", 0, 90, 0)
        with c_col2:
            crop_bottom = st.slider("Bottom (%)", 10, 100, 100)
            crop_right = st.slider("Right (%)", 10, 100, 100)

    if st.button("🔍 Recognize Digit from Photo", type="primary"):
        with st.spinner("Processing image and predicting..."):
            # Determine region to process
            if use_manual_crop and (crop_top < crop_bottom and crop_left < crop_right):
                y1 = int(h_full * crop_top / 100)
                y2 = int(h_full * crop_bottom / 100)
                x1 = int(w_full * crop_left / 100)
                x2 = int(w_full * crop_right / 100)
                proc_img = rgb_full[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                proc_img = rgb_full
                offset_x, offset_y = 0, 0

            canvas, bbox, debug_info, err_msg = extract_digit_from_photo(proc_img)

            if err_msg or canvas is None:
                st.error(f"Detection error: {err_msg}")
            else:
                # Prepare CNN input
                img_array = (canvas.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_digit = int(np.argmax(predictions))
                model_score = float(predictions[predicted_digit]) * 100

                # Display detected bounding box on original image
                bx, by, bw, bh = bbox
                annotated = rgb_full.copy()
                global_bx = bx + offset_x
                global_by = by + offset_y
                cv2.rectangle(
                    annotated,
                    (global_bx, global_by),
                    (global_bx + bw, global_by + bh),
                    (0, 220, 0),
                    max(2, int(min(h_full, w_full) * 0.005))
                )

                st.subheader("🎯 Detection & Prediction")
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.image(
                        annotated,
                        caption="Automatically Detected Digit (Green Box)",
                        use_container_width=True
                    )

                with res_col2:
                    st.image(
                        canvas,
                        caption="Processed 28 × 28 CNN Input",
                        width=180
                    )
                    st.success(f"### Predicted Digit: **{predicted_digit}**")
                    st.metric("Model Confidence", f"{model_score:.2f}%")

                # Probability bar chart
                st.subheader("📊 Confidence Distribution (Digits 0–9)")
                score_data = {str(i): float(predictions[i]) for i in range(10)}
                st.bar_chart(score_data)

                # Debug pipeline
                with st.expander("🛠️ View Image Preprocessing Steps"):
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.image(debug_info["diff"], caption="1. Illumination Normalized (Ink Difference)")
                        st.image(debug_info["line_mask"], caption="2. Detected Notebook Ruled Lines")
                    with d_col2:
                        st.image(debug_info["stroke_bridge"], caption="3. Cleaned & Stroke-Bridged Mask")
                        st.image(debug_info["digit_mask"], caption="4. Isolated Digit Mask")
                    st.image(canvas, caption="5. Final 28×28 Centered Input", width=140)


# ==================================================
# 6. DRAW DIGIT ON SCREEN SECTION
# ==================================================

st.divider()
st.header("✍️ Draw a Digit")
st.write("Draw any digit from 0 to 9 in the canvas below:")

if "canvas_reset_counter" not in st.session_state:
    st.session_state.canvas_reset_counter = 0

if CANVAS_AVAILABLE:
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🧹 Clear Canvas"):
            st.session_state.canvas_reset_counter += 1
            st.rerun()

    canvas_result = st_canvas(
        stroke_width=16,
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

            pts = cv2.findNonZero(binary)
            if pts is None:
                st.warning("Canvas is blank. Please draw a digit.")
            else:
                x, y, w, h = cv2.boundingRect(pts)
                digit = binary[y:y + h, x:x + w]

                # Pad
                padding = max(4, int(max(w, h) * 0.15))
                padded = cv2.copyMakeBorder(
                    digit, padding, padding, padding, padding,
                    cv2.BORDER_CONSTANT, value=0
                )

                # Resize to fit 20x20
                scale = 20.0 / max(padded.shape)
                new_w = max(1, int(round(padded.shape[1] * scale)))
                new_h = max(1, int(round(padded.shape[0] * scale)))
                resized = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Center in 28x28
                canvas = np.zeros((28, 28), dtype=np.uint8)
                ox = (28 - new_w) // 2
                oy = (28 - new_h) // 2
                canvas[oy:oy + new_h, ox:ox + new_w] = resized

                # Center of mass alignment
                canvas = center_by_center_of_mass(canvas)

                # Predict
                img_array = (canvas.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_digit = int(np.argmax(predictions))
                model_score = float(predictions[predicted_digit]) * 100

                st.subheader("🎯 Drawn Digit Result")
                draw_col1, draw_col2 = st.columns([1, 2])
                with draw_col1:
                    st.image(canvas, caption="28 × 28 CNN Input", width=160)
                with draw_col2:
                    st.success(f"### Predicted Digit: **{predicted_digit}**")
                    st.metric("Model Confidence", f"{model_score:.2f}%")

                st.subheader("📊 Confidence Distribution (Digits 0–9)")
                score_data = {str(i): float(predictions[i]) for i in range(10)}
                st.bar_chart(score_data)
else:
    st.info("Interactive canvas drawing is disabled in this environment. You can upload photo images above!")
