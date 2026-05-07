import cv2
import streamlit as st
import numpy as np
import time
import os
import urllib.request
import json
from streamlit_option_menu import option_menu

@st.cache_data
def get_location():
    try:
        req = urllib.request.Request("https://ipapi.co/json/", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            city = data.get("city", "Unknown")
            country = data.get("country_name", "Unknown")
            if city != "Unknown":
                return f"{city}, {country}"
            return "Unknown"
    except Exception:
        return "Unknown"

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="VisionAI | Object & Emotion Tracker",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"], p, span, div, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF !important;
    }
    
    .stApp { 
        background: radial-gradient(circle at 10% 20%, rgba(14, 17, 23, 1) 0%, rgba(25, 28, 36, 1) 100%); 
        color: #FFFFFF; 
    }
    
    /* Clean up defaults */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animated Main Title */
    .main-title {
        font-size: 3.5rem; 
        font-weight: 800;
        background: linear-gradient(-45deg, #FF3CAC, #784BA0, #2B86C5, #00C9FF);
        background-size: 300% 300%;
        animation: gradient-animation 8s ease infinite;
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent !important;
        margin-bottom: -15px;
        letter-spacing: -1px;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-title { 
        font-size: 1.25rem; 
        color: #94A3B8; 
        margin-bottom: 30px; 
        font-weight: 300;
    }
    
    /* Premium Glassmorphism Metrics */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px; 
        border-radius: 16px; 
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 201, 255, 0.15);
        border: 1px solid rgba(0, 201, 255, 0.3);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(14, 17, 23, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Content styling */
    h3 {
        color: #E2E8F0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Load Models (Face + YOLO Objects)
# ==========================================
@st.cache_resource
def load_cascades():
    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face_casc, eye_casc, smile_casc

@st.cache_resource(show_spinner="Downloading Object Detection Model (YOLOv4-tiny, 24MB)... Please wait, this happens only once!")
def load_yolo():
    cfg_path = "yolov4-tiny.cfg"
    weights_path = "yolov4-tiny.weights"
    
    # Auto-download the lightweight YOLO model so the user doesn't have to do any complex setup
    if not os.path.exists(cfg_path):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg", cfg_path)
    if not os.path.exists(weights_path):
        urllib.request.urlretrieve("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights", weights_path)
        
    net = cv2.dnn.readNet(weights_path, cfg_path)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    
    classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    return model, classes

face_cascade, eye_cascade, smile_cascade = load_cascades()

EMOJI_MAP = { "Neutral": "😐", "Happy": "😊" }

def detect_emotion(gray_face):
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=20)
    if len(smiles) > 0: return "Happy"
    return "Neutral"

# ==========================================
# 3. Control Panel (Moved to Main UI)
# ==========================================

# ==========================================
# 4. Main Page UI
# ==========================================
st.markdown('<p class="main-title">VisionAI Tracker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Facial & 80-Class Object Detection Dashboard</p>', unsafe_allow_html=True)

selected_menu = option_menu(
    menu_title=None,
    options=["Dashboard", "Analytics", "Models", "Settings"],
    icons=["house", "graph-up", "box", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important", 
            "background-color": "rgba(255, 255, 255, 0.03)", 
            "border-radius": "16px", 
            "border": "1px solid rgba(255, 255, 255, 0.1)",
            "margin-bottom": "20px",
            "backdrop-filter": "blur(10px)"
        },
        "icon": {"color": "#00C9FF", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px", 
            "text-align": "center", 
            "margin": "0px", 
            "--hover-color": "rgba(0, 201, 255, 0.1)",
            "color": "#FAFAFA",
            "font-family": "'Outfit', sans-serif"
        },
        "nav-link-selected": {
            "background-color": "rgba(0, 201, 255, 0.2)", 
            "border": "1px solid rgba(0, 201, 255, 0.3)",
            "font-weight": "600"
        },
    }
)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1: fps_metric = st.empty()
with col2: face_metric = st.empty()
with col3: emotion_metric = st.empty()
with col4: object_metric = st.empty()
with col5: acc_metric = st.empty()
with col6: loc_metric = st.empty()

user_location = get_location()

fps_metric.metric("⚡ Current FPS", "0")
face_metric.metric("👤 Faces", "0")
emotion_metric.metric("🎭 Emotion", "None")
object_metric.metric("📦 Objects", "0")
acc_metric.metric("🎯 Avg Accuracy", "0%")
loc_metric.metric("🌍 Location", user_location)

st.markdown("---")

col_cam, col_spacer = st.columns([2.5, 1])
with col_cam: FRAME_WINDOW = st.empty()
with col_spacer:
    st.markdown("### ⚙️ Control Panel")
    
    input_mode = st.radio("📥 Select Input Mode:", ["Live Webcam", "Upload Photo", "Take Photo"])
    run = False
    uploaded_file = None
    if input_mode == "Live Webcam":
        try: run = st.toggle("🎥 Enable Live Camera", value=False)
        except AttributeError: run = st.checkbox("🎥 Enable Live Camera", value=False)
    elif input_mode == "Upload Photo":
        uploaded_file = st.file_uploader("📤 Upload an Image", type=['jpg', 'jpeg', 'png'])
    else:
        uploaded_file = st.camera_input("📸 Take a Picture")
        
    st.markdown("### 🎛️ Detection Features")
    detect_objects = st.checkbox("🔍 Detect All Objects", value=True)
    draw_faces = st.checkbox("Show Face & Emotion", value=True)
    draw_eyes = st.checkbox("Show Eyes", value=True)
    draw_smile = st.checkbox("Show Smile", value=True)
    draw_threshold = st.checkbox("➖ Draw Threshold Line", value=False)
    if draw_threshold:
        threshold_y = st.slider("Line Position (%)", 10, 90, 50)
    else:
        threshold_y = 50
    
    st.info("YOLOv4-tiny object detection active!")
    
    st.markdown("### 🏷️ Detected Objects")
    objects_list_placeholder = st.empty()

# Initialize YOLO model only if enabled and running to save memory
yolo_model = None
yolo_classes = []
if detect_objects and (run or uploaded_file):
    yolo_model, yolo_classes = load_yolo()

# ==========================================
# 5. Core Processing Logic
# ==========================================
def process_frame(frame):
    num_faces = 0
    dominant_emotion = "Neutral"
    num_objects = 0
    avg_accuracy = 0
    detected_names = []
    
    # 1. Object Detection (YOLO 80-class)
    if detect_objects and yolo_model is not None:
        classIds, scores, boxes = yolo_model.detect(frame, confThreshold=0.4, nmsThreshold=0.4)
        
        if len(classIds) > 0:
            # Flatten arrays to handle different OpenCV version return structures safely
            classIds = classIds.flatten() if hasattr(classIds, 'flatten') else classIds
            scores = scores.flatten() if hasattr(scores, 'flatten') else scores
            
            num_objects = len(classIds)
            avg_accuracy = int(np.mean(scores) * 100) if num_objects > 0 else 0
            
            for (classId, score, box) in zip(classIds, scores, boxes):
                label_name = yolo_classes[classId]
                detected_names.append(label_name)
                
                # Draw pinkish bounding box for general objects
                color = (255, 100, 200) 
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                
                label_text = f" {label_name} {int(score * 100)}% "
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                
                # Solid background badge for object label
                cv2.rectangle(frame, (box[0], box[1] - text_height - 5), (box[0] + text_width, box[1]), color, cv2.FILLED)
                cv2.putText(frame, label_text, (box[0], box[1] - 3), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # 2. Face & Emotion Detection (Haar Cascades)
    if draw_faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        num_faces = len(faces)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 120), 3)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            if draw_eyes:
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 150, 0), 2)
                    
            if draw_smile:
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                    
            emotion = detect_emotion(roi_gray)
            dominant_emotion = emotion
            
            # Estimate confidence based on face box size (since Haar doesn't natively return % probabilities easily)
            face_acc = min(99, max(75, int(70 + (w * h) / 2000)))
            label = f" Face: {emotion} {face_acc}% "
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            
            # Write mood INSIDE face box
            text_x = x
            text_y = y + text_height + 10
            
            cv2.rectangle(frame, (text_x, y), (text_x + text_width, text_y), (0, 255, 120), cv2.FILLED)
            cv2.putText(frame, label, (text_x, text_y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
    if draw_threshold:
        h, w = frame.shape[:2]
        line_y = int(h * (threshold_y / 100))
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 3)
        cv2.putText(frame, "Threshold", (10, line_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            
    return frame, num_faces, dominant_emotion, num_objects, avg_accuracy, detected_names

# ==========================================
# 6. Run Mode Logic (Live / Photo)
# ==========================================
if input_mode == "Live Webcam":
    if not run:
        FRAME_WINDOW.info("⏸️ Camera is paused. Toggle 'Enable Live Camera' in the sidebar to start.")
    else:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("🚨 Error: Could not access the webcam.")
        else:
            prev_time = time.time()
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture video feed.")
                    break
                    
                frame = cv2.resize(frame, (800, 600))
                processed_frame, num_faces, dom_emotion, num_objects, avg_acc, detected_names = process_frame(frame)
                
                if detected_names:
                    from collections import Counter
                    counts = Counter(detected_names)
                    list_text = "\n".join([f"- **{name.title()}** (x{count})" for name, count in counts.items()])
                    objects_list_placeholder.markdown(list_text)
                else:
                    objects_list_placeholder.markdown("_None_")
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                fps_metric.metric("⚡ Current FPS", f"{int(fps)}")
                face_metric.metric("👤 Faces", f"{num_faces}")
                emotion_metric.metric("🎭 Emotion", f"{dom_emotion} {EMOJI_MAP.get(dom_emotion, '')}")
                object_metric.metric("📦 Objects", f"{num_objects}")
                acc_metric.metric("🎯 Avg Accuracy", f"{avg_acc}%" if avg_acc > 0 else "0%")
                
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                try: FRAME_WINDOW.image(frame_rgb, use_container_width=True)
                except TypeError: FRAME_WINDOW.image(frame_rgb)
                
            camera.release()

elif input_mode in ["Upload Photo", "Take Photo"]:
    if uploaded_file is None:
        FRAME_WINDOW.info("🖼️ Please provide a photo from the sidebar on the left.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        if frame is not None:
            h, w = frame.shape[:2]
            if w > 800:
                ratio = 800 / w
                frame = cv2.resize(frame, (800, int(h * ratio)))
            
            processed_frame, num_faces, dom_emotion, num_objects, avg_acc, detected_names = process_frame(frame)
            
            if detected_names:
                from collections import Counter
                counts = Counter(detected_names)
                list_text = "\n".join([f"- **{name.title()}** (x{count})" for name, count in counts.items()])
                objects_list_placeholder.markdown(list_text)
            else:
                objects_list_placeholder.markdown("_None_")
            
            fps_metric.metric("⚡ Current FPS", "N/A (Photo)")
            face_metric.metric("👤 Faces", f"{num_faces}")
            emotion_metric.metric("🎭 Emotion", f"{dom_emotion} {EMOJI_MAP.get(dom_emotion, '')}")
            object_metric.metric("📦 Objects", f"{num_objects}")
            acc_metric.metric("🎯 Avg Accuracy", f"{avg_acc}%" if avg_acc > 0 else "0%")
            
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            try: FRAME_WINDOW.image(frame_rgb, use_container_width=True)
            except TypeError: FRAME_WINDOW.image(frame_rgb)
        else:
            st.error("Failed to process the uploaded image. Please try a different photo.")