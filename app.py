import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

st.set_page_config(page_title="Image Object Detection", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); color: #e8eaf6; }
#MainMenu, footer { visibility: hidden; }
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero h1 { font-size: 2.8rem; font-weight: 700; background: linear-gradient(90deg, #00d2ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #9fa8da; font-size: 1.05rem; }
[data-testid="stFileUploader"] { background: rgba(255,255,255,0.04); border: 2px dashed rgba(0,210,255,0.35); border-radius: 14px; padding: 1.2rem; }
[data-testid="stFileUploaderDropzoneInput"] + div button,
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 1.1rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(30, 136, 229, 0.45) !important;
}
[data-testid="stFileUploaderDropzoneInput"] + div button:hover,
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #1976d2, #42a5f5) !important;
    box-shadow: 0 6px 20px rgba(30, 136, 229, 0.65) !important;
    transform: translateY(-2px) !important;
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzoneInput"] + div button:active,
[data-testid="stFileUploader"] button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(30, 136, 229, 0.4) !important;
}
[data-testid="metric-container"] { background: rgba(255,255,255,0.06); border: 1px solid rgba(0,210,255,0.2); border-radius: 12px; padding: 1rem; text-align: center; }
[data-testid="stMetricLabel"] { color: #9fa8da !important; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #00d2ff !important; font-size: 2rem; font-weight: 700; }
.img-label { font-weight: 600; color: #9fa8da; margin-bottom: 0.3rem; font-size: 0.95rem; }
.det-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
.det-table th { background: rgba(123,47,247,0.3); color: #e8eaf6; padding: 0.6rem 1rem; text-align: left; }
.det-table td { padding: 0.5rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.07); color: #cfd8dc; }
.det-table tr:hover td { background: rgba(255,255,255,0.05); }
.conf-track { background: rgba(255,255,255,0.1); border-radius: 20px; height: 10px; width: 120px; }
.conf-fill { background: linear-gradient(90deg, #00d2ff, #7b2ff7); border-radius: 20px; height: 10px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.chip { background: rgba(0,210,255,0.12); border: 1px solid rgba(0,210,255,0.4); color: #00d2ff; border-radius: 20px; padding: 0.25rem 0.85rem; font-size: 0.82rem; }
hr { border-color: rgba(255,255,255,0.08) !important; }
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #9b51e0);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(123, 47, 247, 0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9b51e0, #c084fc);
    box-shadow: 0 6px 20px rgba(123, 47, 247, 0.65);
    transform: translateY(-2px);
}
.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 8px rgba(123, 47, 247, 0.4);
}
</style>
""", unsafe_allow_html=True)

COLORS = [
    (0, 212, 255), (255, 100, 0), (50, 255, 130), (255, 50, 160),
    (160, 50, 255), (255, 220, 0), (0, 160, 255), (255, 80, 80),
]

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def get_color(class_id):
    return COLORS[class_id % len(COLORS)]

def draw_boxes(image, results):
    annotated = image.copy()
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            name = model.names[class_id]
            color = get_color(class_id)

            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            L = 16
            for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(annotated, (px, py), (px + dx*L, py), color, 3)
                cv2.line(annotated, (px, py), (px, py + dy*L), color, 3)

            label = f"{name}  {conf:.0%}"
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
            pad = 4
            ly1 = max(y1 - th - 2*pad, 0)
            ly2 = ly1 + th + 2*pad
            cv2.rectangle(annotated, (x1, ly1), (x1 + tw + 2*pad, ly2), color, -1)
            cv2.putText(annotated, label, (x1+pad, ly2-pad), font, 0.55, (15,15,15), 1, cv2.LINE_AA)

            detections.append({"name": name, "conf": conf})

    return annotated, detections

st.markdown('<div class="hero"><h1>🔍 Image Object Detection</h1><p>Upload an image · AI detects every object · Powered by YOLOv8</p></div>', unsafe_allow_html=True)
st.markdown("---")



uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem;background:rgba(255,255,255,0.03);border:2px dashed rgba(0,210,255,0.25);border-radius:16px;margin-top:1rem;">
        <div style="font-size:3.5rem;">📂</div>
        <p style="color:#9fa8da;font-size:1.1rem;">Upload a JPG, JPEG, or PNG to get started</p>
        <p style="color:#546e7a;font-size:0.85rem;">Processed locally — nothing is uploaded to any server.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if bgr is None:
    st.error("❌ Could not read image. Please upload a valid JPG/JPEG/PNG.")
    st.stop()

original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

with st.spinner("🔍 Detecting objects..."):
    results = model(bgr, conf=0.40, verbose=False)
    annotated_bgr, detections = draw_boxes(bgr, results)

annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

total = len(detections)
unique = list({d["name"] for d in detections})
avg_conf = sum(d["conf"] for d in detections) / total if total > 0 else 0.0

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("🎯 Objects Detected", total)
c2.metric("🏷️ Unique Classes", len(unique))
c3.metric("📊 Avg Confidence", f"{avg_conf:.0%}")
st.markdown("---")

left, right = st.columns(2, gap="large")
with left:
    st.markdown('<p class="img-label">📷 Original Image</p>', unsafe_allow_html=True)
    st.image(original_rgb, use_container_width=True)
with right:
    st.markdown('<p class="img-label">🤖 Detected Objects</p>', unsafe_allow_html=True)
    st.image(annotated_rgb, use_container_width=True)

st.markdown("---")
st.markdown('<p style="font-size:1.1rem;font-weight:600;color:#7b2ff7;text-transform:uppercase;letter-spacing:0.08em;">📋 Detection Results</p>', unsafe_allow_html=True)

if detections:
    rows = ""
    for i, d in enumerate(detections, 1):
        bar = int(d["conf"] * 100)
        rows += f"""
        <tr>
            <td>#{i}</td>
            <td><strong>{d['name'].title()}</strong></td>
            <td><div class="conf-track"><div class="conf-fill" style="width:{bar}%;"></div></div></td>
            <td>{d['conf']:.1%}</td>
        </tr>"""
    st.markdown(f"""
    <table class="det-table">
        <thead><tr><th>#</th><th>Object</th><th>Confidence Bar</th><th>Score</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("---")
    chips = "".join(f'<span class="chip">{c.title()}</span>' for c in sorted(unique))
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
else:
    st.info("No objects detected. Try lowering the confidence threshold.", icon="💡")
