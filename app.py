# =============================================================
#  app.py  —  Image Object Detection  |  YOLOv8 + Streamlit
# =============================================================
#
#  HOW THIS FILE WORKS (plain English):
#  1. The user opens the app in a browser (via `streamlit run app.py`).
#  2. They upload a JPG / JPEG / PNG image.
#  3. We convert the image so OpenCV can read it.
#  4. We pass it through the YOLOv8n model (tiny, free, pretrained).
#  5. The model tells us: "I see a person at these coordinates
#     with 91% confidence."
#  6. We draw a coloured box + label on the image.
#  7. We show the original and the detected image side by side.
#  8. We list all detections in a table below.
# =============================================================

import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2


# -------------------------------------------------------------
# PAGE CONFIGURATION
# Must be the FIRST Streamlit call in the script.
# -------------------------------------------------------------
st.set_page_config(
    page_title="Image Object Detection",
    page_icon="🔍",
    layout="wide",
)


# -------------------------------------------------------------
# CUSTOM CSS  —  makes the app look clean and modern
# -------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e8eaf6;
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #9fa8da;
    font-size: 1.05rem;
}

/* ── Upload box area ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(0, 210, 255, 0.35);
    border-radius: 14px;
    padding: 1.2rem;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 210, 255, 0.75);
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(0, 210, 255, 0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
[data-testid="stMetricLabel"]  { color: #9fa8da !important; font-size: 0.85rem; }
[data-testid="stMetricValue"]  { color: #00d2ff !important; font-size: 2rem; font-weight: 700; }

/* ── Section headings ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7b2ff7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* ── Detection results table ── */
.det-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}
.det-table th {
    background: rgba(123, 47, 247, 0.3);
    color: #e8eaf6;
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 600;
}
.det-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    color: #cfd8dc;
}
.det-table tr:hover td { background: rgba(255,255,255,0.05); }

/* ── Confidence progress bar ── */
.conf-track {
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    height: 10px;
    width: 120px;
}
.conf-fill {
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    border-radius: 20px;
    height: 10px;
}

/* ── Class chip badges ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.chip {
    background: rgba(0, 210, 255, 0.12);
    border: 1px solid rgba(0, 210, 255, 0.4);
    color: #00d2ff;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── No-detection message ── */
.no-det {
    text-align: center;
    padding: 2rem;
    color: #9fa8da;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
}

/* ── Image caption labels ── */
.img-label {
    font-weight: 600;
    color: #9fa8da;
    margin-bottom: 0.3rem;
    font-size: 0.95rem;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# LOAD MODEL  (cached so it only loads once per session)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading YOLOv8 model…")
def load_model():
    """
    Load the YOLOv8 nano model.
    The first time this runs, it downloads yolov8n.pt (~6 MB).
    After that it's cached locally — no internet needed.
    """
    return YOLO("yolov8n.pt")   # 'n' = nano (smallest & fastest)


model = load_model()


# -------------------------------------------------------------
# COLOUR PALETTE  —  one distinct colour per object class
# -------------------------------------------------------------
COLORS = [
    (0,   212, 255),   # cyan
    (255, 100,   0),   # orange
    (50,  255, 130),   # green
    (255,  50, 160),   # pink
    (160,  50, 255),   # purple
    (255, 220,   0),   # yellow
    (0,   160, 255),   # sky-blue
    (255,  80,  80),   # red
    (80,  255,  80),   # lime
    (255, 160,  80),   # gold
]

def get_color(class_id: int) -> tuple:
    """Return a BGR colour that is consistent for a given class."""
    return COLORS[class_id % len(COLORS)]


# -------------------------------------------------------------
# DRAW DETECTIONS  —  called on the BGR NumPy array
# -------------------------------------------------------------
def draw_detections(image: np.ndarray, results) -> tuple:
    """
    Draw bounding boxes and labels on the image.

    Returns
    -------
    annotated : np.ndarray  — image with boxes drawn
    detections : list[dict] — list of {class_name, confidence, bbox}
    """
    annotated  = image.copy()
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # ── Coordinates ──
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf            = float(box.conf[0])
            class_id        = int(box.cls[0])
            class_name      = model.names[class_id]
            color           = get_color(class_id)

            # ── Semi-transparent fill ──
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)

            # ── Border ──
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # ── Futuristic corner accents ──
            L, T = 16, 3          # corner length, thickness
            for (px, py, dx, dy) in [
                (x1, y1, +1, +1),   # top-left
                (x2, y1, -1, +1),   # top-right
                (x1, y2, +1, -1),   # bottom-left
                (x2, y2, -1, -1),   # bottom-right
            ]:
                cv2.line(annotated, (px, py), (px + dx * L, py), color, T)
                cv2.line(annotated, (px, py), (px, py + dy * L), color, T)

            # ── Label pill ──
            label          = f"{class_name}  {conf:.0%}"
            font           = cv2.FONT_HERSHEY_DUPLEX
            font_scale     = 0.55
            font_thickness = 1
            (tw, th), _    = cv2.getTextSize(label, font, font_scale, font_thickness)
            pad = 4
            lbl_y1 = max(y1 - th - 2 * pad, 0)
            lbl_y2 = lbl_y1 + th + 2 * pad
            cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 2 * pad, lbl_y2), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + pad, lbl_y2 - pad),
                font, font_scale, (15, 15, 15), font_thickness, cv2.LINE_AA,
            )

            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })

    return annotated, detections


# =============================================================
#  UI  LAYOUT
# =============================================================

# ── Hero banner ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Image Object Detection</h1>
    <p>Upload any image · AI finds every object · Powered by YOLOv8</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # Confidence threshold slider
    confidence_threshold = st.slider(
        label="🎯 Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05,
        help=(
            "Only show detections above this confidence score.\n\n"
            "• Lower → detects more objects (may include false positives)\n"
            "• Higher → only very confident detections"
        ),
    )

    st.markdown("---")
    st.markdown("""
    ### 📌 How to use
    1. Adjust the confidence slider if needed.  
    2. Upload a JPG / JPEG / PNG image.  
    3. See detected objects instantly!
    """)

    st.markdown("---")
    st.markdown("""
    ### 🤖 About the Model
    **YOLOv8n** (nano) by Ultralytics  
    Trained on **COCO** — 80 object classes  
    Free · Local · No API needed
    """)

# ── File uploader ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop an image here or click Browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ── Waiting for upload ────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:3rem; background:rgba(255,255,255,0.03);
                border:2px dashed rgba(0,210,255,0.25); border-radius:16px; margin-top:1rem;">
        <div style="font-size:3.5rem;">📂</div>
        <p style="color:#9fa8da; font-size:1.1rem; margin:0.5rem 0;">
            Upload a JPG, JPEG, or PNG to get started
        </p>
        <p style="color:#546e7a; font-size:0.85rem;">
            Your image is processed locally — nothing is sent to any server.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # stop script here — nothing more to render yet


# ── Convert uploaded file → NumPy (OpenCV) ──────────────────
# Read raw bytes from the upload, decode directly with OpenCV.
# cv2.imdecode expects a NumPy array of bytes and returns a BGR image.
file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
bgr_image  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)   # BGR NumPy array

if bgr_image is None:
    st.error("❌ Could not read the image. Please upload a valid JPG/JPEG/PNG file.")
    st.stop()

# Convert BGR → RGB once — used only for the 'Original Image' display
original_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


# ── Run YOLOv8 detection ──────────────────────────────────────
with st.spinner("🔍 Detecting objects…"):
    results = model(bgr_image, conf=confidence_threshold, verbose=False)
    annotated_bgr, detections = draw_detections(bgr_image, results)

# Convert annotated BGR image back to RGB so Streamlit can display it
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


# ── Summary Metrics ───────────────────────────────────────────
st.markdown("---")
total_objects   = len(detections)
unique_classes  = list({d["class_name"] for d in detections})
avg_confidence  = (
    sum(d["confidence"] for d in detections) / total_objects
    if total_objects > 0 else 0.0
)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("🎯 Objects Detected", total_objects)
col_m2.metric("🏷️ Unique Classes",  len(unique_classes))
col_m3.metric("📊 Avg Confidence",  f"{avg_confidence:.0%}")

st.markdown("---")


# ── Side-by-side image display ────────────────────────────────
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown('<p class="img-label">📷 Original Image</p>', unsafe_allow_html=True)
    st.image(original_rgb, use_container_width=True)

with right_col:
    st.markdown('<p class="img-label">🤖 Detected Objects</p>', unsafe_allow_html=True)
    st.image(annotated_rgb, use_container_width=True)

st.markdown("---")


# ── Detection Results Table ───────────────────────────────────
st.markdown('<p class="section-title">📋 Detection Results</p>', unsafe_allow_html=True)

if detections:
    # Build table rows as HTML for custom styling
    rows_html = ""
    for i, det in enumerate(detections, start=1):
        bar_w = int(det["confidence"] * 100)
        rows_html += f"""
        <tr>
            <td>#{i}</td>
            <td><strong>{det['class_name'].title()}</strong></td>
            <td>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{bar_w}%;"></div>
                </div>
            </td>
            <td>{det['confidence']:.1%}</td>
        </tr>"""

    st.markdown(f"""
    <table class="det-table">
        <thead>
            <tr><th>#</th><th>Object</th><th>Confidence Bar</th><th>Score</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Class chips
    st.markdown("---")
    st.markdown('<p class="section-title">🏷️ Classes Found</p>', unsafe_allow_html=True)
    chips = "".join(f'<span class="chip">{c.title()}</span>' for c in sorted(unique_classes))
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

else:
    # Nothing detected
    st.markdown("""
    <div class="no-det">
        <div style="font-size:2.5rem;">🤷</div>
        <p>No objects detected at this confidence level.</p>
        <p style="font-size:0.85rem; color:#546e7a;">
            Try lowering the <strong>Confidence Threshold</strong> in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
