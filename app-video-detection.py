import streamlit as st
import cv2
import tempfile
import time
import os
from roboflow import Roboflow
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Inisialisasi Roboflow
API_KEY = "v5uSkL96TA5w3aVKvls8"
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("fire-and-smoke-volcano")
model = project.version(1).model

# Gambar bounding box
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

# Konfigurasi halaman
st.set_page_config(page_title="🔥 Volcano Detection", layout="wide")
st.title("🔥 Smoke/Fire Detection on Volcano Surveillance Video")

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

uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi", "mpeg4"])
frame_skip = st.slider("⏩ Process every Nth frame", 1, 30, 5)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())
        input_path = temp_input.name

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = input_path.replace(".mp4", "_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    anomaly_count = 0
    detection_counts = []

    stframe = st.empty()
    progress = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            prediction = model.predict(frame, confidence=40, overlap=30).json()
            preds = prediction.get("predictions", [])
            detection_counts.append(len(preds))

            if preds:
                anomaly_count += 1
            frame = draw_boxes_on_frame(frame, preds)
            stframe.image(frame, channels="BGR", use_container_width=True)

        out.write(frame)
        frame_count += 1
        progress.progress(min(100, int((frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100)))

    cap.release()
    out.release()

    st.success("✅ Video processing complete.")
    st.video(output_path)

    # Unduh video
    with open(output_path, "rb") as file:
        st.download_button("⬇ Download Processed Video", data=file, file_name="processed_video.mp4")

    # Kesimpulan
    st.subheader("🧠 Automated Detection Conclusion")
    if anomaly_count >= 3:
        st.error(f"🚨 Anomaly Detected: This video shows repeated signs of smoke/fire indicating possible volcanic activity.")
    else:
        st.info(f"✅ Normal Activity Detected: Only {anomaly_count} frames showed anomaly.")

    # Grafik
    st.subheader("📊 Detection Count per Frame")
    fig, ax = plt.subplots()
    ax.bar(range(len(detection_counts)), detection_counts, color="red")
    ax.set_xlabel("Frame Sample")
    ax.set_ylabel("Detection Count")
    ax.set_title("Detections per Processed Frame")
    st.pyplot(fig)
