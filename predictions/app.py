import streamlit as st
import cv2
import numpy as np
import time
import os
import psycopg2
from datetime import datetime
from helper import YOLO_Pred
from twilio.rest import Client
from dotenv import load_dotenv

# Load the .env file in the same folder
#(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# === Initialize YOLO ===
yolo = YOLO_Pred('predictions/hell/weights/best.onnx', 'predictions/data.yaml')

# === Twilio Configuration ===
TWILIO_ACCOUNT_SID = 'AC32ba2e1cc6874969ec6c6687b9937900'
TWILIO_AUTH_TOKEN = 'ec1f85b3033145a4312cbf3be075047c'
TWILIO_PHONE_NUMBER = '+17152266011' 
RECIPIENT_PHONE_NUMBER = '+916206920880'

# === PostgreSQL Setup ===
DB_URL = 'postgresql://nitin_user:mxigVvzicOhcat6mGn5ltTHk7062VnJt@dpg-cvshe2i4d50c738h9a30-a.oregon-postgres.render.com/nitin'

def init_db():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            timestamp TEXT,
            label TEXT,
            confidence REAL,
            distance_m REAL,
            source TEXT
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def log_detection(label, confidence, distance, source):
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (timestamp, label, confidence, distance_m, source)
        VALUES (%s, %s, %s, %s, %s)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence, round(distance, 2), source))
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# === Twilio SMS Function ===
def send_sms(to_phone, message_body):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        st.success(f"📱 SMS sent to {to_phone}: {message.sid}")
    except Exception as e:
        st.error(f"❌ Failed to send SMS: {e}")

# === Distance Calculation ===
def calculate_distance(bbox_width, focal_length=800, real_width=2.5):
    if bbox_width > 0:
        return (focal_length * real_width) / bbox_width
    return float('inf')

# === Check if Vehicle is New ===
def is_new_vehicle(cx, cy, existing_coords, threshold=50):
    for ex, ey in existing_coords:
        if abs(cx - ex) < threshold and abs(cy - ey) < threshold:
            return False
    return True

# === Draw Multi-Line Label ===
def draw_label(img, label_texts, x, y, frame_width):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = frame_width / 800  # Slightly increased
    thickness = 2
    text_heights = []
    max_text_width = 0
    for text in label_texts:
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_heights.append(text_h)
        max_text_width = max(max_text_width, text_w)
    total_height = sum(text_heights) + 5 * len(label_texts)
    padding = 10
    img_h, img_w = img.shape[:2]
    if y - total_height - padding < 0:
        y_text_start = y + padding + 5
        box_top = y
        box_bottom = y_text_start + total_height
    else:
        y_text_start = y - total_height - padding
        box_top = y_text_start
        box_bottom = y
    box_left = max(x, 0)
    box_right = min(x + max_text_width + 10, img_w)
    cv2.rectangle(img, (box_left, box_top), (box_right, box_bottom), (0, 0, 0), -1)
    current_y = y_text_start + 5
    for i, text in enumerate(label_texts):
        cv2.putText(img, text, (x + 5, current_y + text_heights[i]), font, font_scale, (255, 255, 255), thickness)
        current_y += text_heights[i] + 5
    return img

# === Streamlit Config ===
st.set_page_config(page_title="🚨 Emergency Vehicle Detection", layout="wide")
st.markdown("""<style>
    body, .main, .stTabs [data-baseweb="tab-list"], .stTabs [data-baseweb="tab"] {
        background-color: #000 !important; color: white !important;
    }
</style>""", unsafe_allow_html=True)

st.title("🚨 Emergency Vehicle Detection System")
st.markdown("Upload a video or image to *detect emergency* and *non-emergency vehicles* in real-time using YOLO and receive SMS alerts via *Twilio*.")

tab1, tab2 = st.tabs(["📹 Video Detection", "🖼 Image Detection"])

# === VIDEO DETECTION ===
with tab1:
    st.subheader("📹 Video Object Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success("✅ Video uploaded successfully!")
        video_file = cv2.VideoCapture(temp_video_path)
        video_placeholder = st.empty()
        message_placeholder = st.empty()
        display_width = 640
        display_height = 360
        fps = video_file.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03
        detected_vehicle_centers = []
        message_sent = False

        while True:
            ret, frame = video_file.read()
            if not ret:
                message_placeholder.success("✅ Video processing completed.")
                break
            img_pred, predicted_texts, boxes = yolo.predictions(frame)
            frame_width = frame.shape[1]
            img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
            new_alert = None
            for i, text in enumerate(predicted_texts):
                x, y, w, h = boxes[i]
                try:
                    label, _ = text.split(' : ')
                except:
                    continue
                label_clean = label.strip().lower()
                cx, cy = x + w // 2, y + h // 2
                img_pred_rgb = draw_label(img_pred_rgb, [label_clean], x, y, frame_width)
                if label_clean == 'emergency':
                    distance = calculate_distance(w)
                    if is_new_vehicle(cx, cy, detected_vehicle_centers):
                        detected_vehicle_centers.append((cx, cy))
                        new_alert = f"🚨 Emergency Vehicle Detected! Distance: {distance:.2f} meters"
                        if not message_sent:
                            send_sms(RECIPIENT_PHONE_NUMBER, new_alert)
                            message_sent = True
                        log_detection(label_clean, 100.0, distance, source="video")
            img_resized = cv2.resize(img_pred_rgb, (display_width, display_height))
            video_placeholder.image(img_resized, caption='🔍 Frame Prediction', use_container_width=True)
            if new_alert:
                message_placeholder.success(new_alert)
            time.sleep(delay)
        video_file.release()

# === IMAGE DETECTION ===
with tab2:
    st.subheader("🖼 Image Object Detection")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        _, predicted_texts, boxes = yolo.predictions(img)
        frame_width = img.shape[1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        message_sent = False
        for i, (text, box) in enumerate(zip(predicted_texts, boxes)):
            x, y, w, h = box
            try:
                label, _ = text.split(' : ')
            except:
                continue
            label_clean = label.strip().lower()
            img_rgb = draw_label(img_rgb, [label_clean], x, y, frame_width)
            if label_clean == 'emergency':
                distance = calculate_distance(w)
                st.success(f"{i + 1}. 🚨 {label_clean} - Distance: {distance:.2f} meters")
                if not message_sent:
                    send_sms(RECIPIENT_PHONE_NUMBER, f"🚨 Emergency Vehicle Detected! Distance: {distance:.2f} meters")
                    message_sent = True
                log_detection(label_clean, 100.0, distance, source="image")
        st.image(img_rgb, caption="🖼 Detected Image", use_container_width=True)

# === Sidebar Info ===
st.sidebar.title("ℹ About")
st.sidebar.info(
    "This app detects emergency vehicles (like ambulances 🚑, fire trucks 🚒, police cars 🚓) in real-time using a YOLO object detection model and alerts the user via Twilio SMS."
)
