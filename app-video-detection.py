import streamlit as st
import cv2
import tempfile
import time
from roboflow import Roboflow
from PIL import Image, ImageDraw
import numpy as np

# Setup model
API_KEY = "v5uSkL96TA5w3aVKvls8"
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("fire-and-smoke-volcano")
model = project.version(1).model

def draw_boxes_on_frame(frame, predictions):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x0 = int(x - w / 2)
        y0 = int(y - h / 2)
        x1 = int(x + w / 2)
        y1 = int(y + h / 2)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, max(0, y0 - 10)), f"{pred['class']} ({pred['confidence']:.2f})", fill="red")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

st.set_page_config(page_title="Volcano Fire/Smoke Detection", layout="wide")
st.title("🔥 Volcano Smoke/Fire Detection App")

st.markdown("""
### 📋 Detection Criteria

**Normal:**
- 0–2 total frames contain fire/smoke
- No repeated detection in consecutive frames
- Only light smoke detected

**Anomaly (Potentially Dangerous):**
- Fire or thick smoke appears in 3+ frames
- Detected in **consecutive or frequent intervals**
- May indicate **pyroclastic flow**, **lava**, or **eruption**
""")

uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi"])
frame_skip = st.slider("⏩ Process every Nth frame", 1, 30, 5)

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(uploaded_video.read())
        video_path = temp_vid.name

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    anomaly_frames = 0

    stframe = st.empty()
    progress = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            _, img_encoded = cv2.imencode(".jpg", frame)
            prediction = model.predict(frame, confidence=40, overlap=30).json()
            predictions = prediction.get("predictions", [])
            frame = draw_boxes_on_frame(frame, predictions)

            if len(predictions) > 0:
                anomaly_frames += 1

            stframe.image(frame, channels="BGR", use_container_width=True)

        frame_count += 1
        progress.progress(min(100, int(frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100)))

    cap.release()

    st.success("✅ Video processing complete!")
    if anomaly_frames >= 3:
        st.error(f"⚠️ Anomaly Detected! ({anomaly_frames} frames)")
    else:
        st.info(f"✅ Normal Activity Detected ({anomaly_frames} frames)")
