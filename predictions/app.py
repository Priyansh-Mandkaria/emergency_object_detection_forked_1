import streamlit as st
import cv2
import numpy as np
import time
import math
import os
from datetime import datetime

from helper import YOLO_Pred
from audio_detector import AudioSirenDetector
from twilio.rest import Client

# === Model Init ===
yolo  = YOLO_Pred("predictions/hell/weights/best.onnx", "predictions/data.yaml")
audio = AudioSirenDetector()

# === Twilio Config (set these as environment variables) ===
TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_PHONE_NUMBER", "+12624259641")

COOLDOWN_SEC = 600  # 10 minutes between SMS to the same number

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
def _init():
    if "locations" not in st.session_state:
        # Default seed locations — edit/remove from the sidebar
        st.session_state.locations = [
            {"name": "Location 1", "x": 0.0,  "y": 50.0, "phone": "+918435932255"},
            {"name": "Location 2", "x": 30.0, "y": 10.0, "phone": "+916205815679"},
        ]
    if "alert_log"   not in st.session_state:
        st.session_state.alert_log = []
    if "sms_sent_at" not in st.session_state:
        st.session_state.sms_sent_at = {}  # phone -> epoch timestamp

_init()

# ──────────────────────────────────────────────
# Pure utility functions
# ──────────────────────────────────────────────

def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _can_sms(phone):
    return time.time() - st.session_state.sms_sent_at.get(phone, 0) > COOLDOWN_SEC


def _send_sms(phone, body):
    if not _can_sms(phone):
        return False
    # Set cooldown immediately so repeated detections don't spam retries
    st.session_state.sms_sent_at[phone] = time.time()
    try:
        Client(TWILIO_SID, TWILIO_TOKEN).messages.create(
            body=body, from_=TWILIO_FROM, to=phone
        )
        return True
    except Exception as e:
        st.error(f"SMS failed → {phone}: {e}")
        return False


def calculate_distance(bbox_w, focal=800, real_w=2.5):
    return (focal * real_w) / bbox_w if bbox_w > 0 else float("inf")


def is_new_vehicle(cx, cy, seen, thr=50):
    return all(abs(cx - x) >= thr or abs(cy - y) >= thr for x, y in seen)


def draw_label(img, lines, x, y, fw):
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 2
    sc    = fw / 800
    sizes = [cv2.getTextSize(t, font, sc, th)[0] for t in lines]
    max_w = max(s[0] for s in sizes)
    hs    = [s[1] for s in sizes]
    total_h = sum(hs) + 5 * len(lines)
    ih, iw  = img.shape[:2]
    y0 = y - total_h - 10 if y - total_h - 10 >= 0 else y + 10
    cv2.rectangle(img, (max(x, 0), y0), (min(x + max_w + 10, iw), y0 + total_h + 5), (0, 0, 0), -1)
    cy_cur = y0 + 5
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + 5, cy_cur + hs[i]), font, sc, (255, 255, 255), th)
        cy_cur += hs[i] + 5
    return img


# ──────────────────────────────────────────────
# Alert routing
# ──────────────────────────────────────────────

def alert_nearest(distance_m, radius_m, audio_ok, source):
    """
    Camera sits at origin (0, 0).
    Emergency vehicle is estimated at (0, distance_m) — directly ahead.

    Strategy:
      • Always notify the NEAREST registered location.
      • Also notify every location within `radius_m` of the vehicle.
      • Respect per-phone cooldown to avoid spam.
    """
    ev_x, ev_y = 0.0, distance_m
    locs = st.session_state.locations
    if not locs:
        return []

    ranked = sorted(locs, key=lambda l: _dist(ev_x, ev_y, l["x"], l["y"]))
    nearest   = ranked[0]
    nearest_d = _dist(ev_x, ev_y, nearest["x"], nearest["y"])

    to_notify = {nearest["phone"]: (nearest, nearest_d, True)}
    for loc in ranked[1:]:
        d = _dist(ev_x, ev_y, loc["x"], loc["y"])
        if d <= radius_m:
            to_notify[loc["phone"]] = (loc, d, False)

    ts  = datetime.now().strftime("%H:%M:%S")
    alerted = []
    for phone, (loc, d, is_nearest) in to_notify.items():
        proximity = "nearest registered point" if is_nearest else f"{d:.0f}m from vehicle"
        body = (
            f"EMERGENCY VEHICLE ALERT [{ts}]\n"
            f"Vehicle ~{distance_m:.0f}m from camera.\n"
            f"You are the {proximity}.\n"
            f"Please clear the road! [{source.upper()}]"
        )
        if _send_sms(phone, body):
            alerted.append(loc["name"])

    st.session_state.alert_log.append({
        "time":    ts,
        "dist":    round(distance_m, 1),
        "audio":   audio_ok,
        "alerted": alerted,
        "source":  source,
    })
    return alerted


# ──────────────────────────────────────────────
# Page layout
# ──────────────────────────────────────────────

st.set_page_config(page_title="Emergency Vehicle Detection", layout="wide")
st.markdown("""<style>
body, .main, .stTabs [data-baseweb="tab-list"], .stTabs [data-baseweb="tab"] {
    background-color:#000 !important; color:white !important;
}
</style>""", unsafe_allow_html=True)

st.title("🚨 Emergency Vehicle Detection System")
st.caption("Simultaneous audio + video detection  ·  Nearest-location SMS routing  ·  Real-time siren analysis")

# ── Sidebar: Location Registry ──────────────────
st.sidebar.title("📍 Location Registry")
st.sidebar.caption(
    "Register locations with coordinates (meters from the camera) "
    "and phone numbers. When an emergency vehicle is detected, the "
    "nearest location—and any within the alert radius—receive an SMS."
)

with st.sidebar.expander("➕ Add Location"):
    n  = st.text_input("Name (e.g. Hospital Gate)", key="new_name")
    px = st.number_input("X — lateral offset (m)",  value=0.0,  key="new_x")
    py = st.number_input("Y — forward distance (m)", value=50.0, key="new_y")
    ph = st.text_input("Phone (+91XXXXXXXXXX)", key="new_phone")
    if st.button("Add Location"):
        if n and ph:
            st.session_state.locations.append(
                {"name": n, "x": px, "y": py, "phone": ph}
            )
            st.rerun()
        else:
            st.warning("Name and phone are required.")

st.sidebar.markdown("**Registered Locations**")
for i, loc in enumerate(list(st.session_state.locations)):
    c1, c2 = st.sidebar.columns([5, 1])
    c1.markdown(
        f"**{loc['name']}**  \n"
        f"({loc['x']:.0f}, {loc['y']:.0f}) m · `{loc['phone']}`"
    )
    if c2.button("✕", key=f"del_{i}"):
        st.session_state.locations.pop(i)
        st.rerun()

st.sidebar.divider()
alert_radius = st.sidebar.slider("Alert radius (m)",          50,   500,  200,  50)
audio_thr    = st.sidebar.slider("Siren detection threshold", 0.05, 0.50, 0.15, 0.01)
require_both = st.sidebar.checkbox(
    "Require BOTH audio + video to trigger alert", value=True,
    help="When enabled, an SMS is only sent if both a visual emergency vehicle "
         "AND siren audio are detected simultaneously."
)

st.sidebar.divider()
st.sidebar.info(
    "Detects ambulances 🚑, fire trucks 🚒, police cars 🚓 using a "
    "YOLOv5 ONNX model + FFT siren-frequency analysis."
)

# ── Main Tabs ───────────────────────────────────
tab1, tab2 = st.tabs(["📹 Video Detection", "🖼 Image Detection"])

# ═══════════════════════════════════════════════
# VIDEO DETECTION
# ═══════════════════════════════════════════════
with tab1:
    st.subheader("📹 Simultaneous Audio + Video Detection")
    st.markdown(
        "Upload a video. The system extracts its audio track and runs **FFT siren analysis** "
        "in sync with YOLO object detection on every frame."
    )
    upload = st.file_uploader("Choose a video…", type=["mp4", "avi", "mov"])

    if upload:
        tmp = "temp_video.mp4"
        with open(tmp, "wb") as f:
            f.write(upload.read())

        with st.spinner("Extracting audio track for siren analysis…"):
            wav_path = audio.extract_audio(tmp)

        if wav_path:
            st.success("🔊 Audio extracted — running parallel audio + video analysis.")
        else:
            st.warning(
                "⚠️ Could not extract audio (ffmpeg not found on PATH). "
                "Falling back to video-only detection."
            )

        cap  = cv2.VideoCapture(tmp)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1.0 / fps

        vid_ph   = st.empty()
        stat_ph  = st.empty()
        audio_ph = st.empty()

        seen_centers  = []
        frame_n       = 0
        sms_sent_this_upload = False  # only 1 SMS per video upload

        while True:
            ok, frame = cap.read()
            if not ok:
                stat_ph.success("✅ Video processing complete.")
                break

            img_pred, texts, boxes = yolo.predictions(frame)
            fw  = frame.shape[1]
            rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            # ── Audio analysis for this frame's timestamp ──
            audio_conf = audio.get_confidence(wav_path, frame_n, fps) if wav_path else 0.0
            audio_ok   = audio_conf >= audio_thr

            if wav_path:
                audio_ph.metric(
                    "🔊 Siren Confidence",
                    f"{audio_conf:.1%}",
                    delta="SIREN DETECTED" if audio_ok else "—",
                )

            # ── Per-detection logic ────────────────────────
            for i, text in enumerate(texts):
                x, y, w, h = boxes[i]
                try:
                    label, conf_str = text.split(" : ")
                except ValueError:
                    continue

                lc = label.strip().lower()
                cx, cy = x + w // 2, y + h // 2

                label_lines = [lc, conf_str.strip()]
                if lc == "emergency" and wav_path:
                    label_lines.append(f"Audio:{audio_conf:.0%}")
                rgb = draw_label(rgb, label_lines, x, y, fw)

                if lc == "emergency":
                    trigger = (not require_both) or audio_ok
                    if trigger and is_new_vehicle(cx, cy, seen_centers):
                        seen_centers.append((cx, cy))
                        d      = calculate_distance(w)
                        source = "video+audio" if audio_ok else "video"
                        msg = f"🚨 Emergency vehicle ~{d:.0f}m away"
                        if not sms_sent_this_upload:
                            alerted = alert_nearest(d, alert_radius, audio_ok, source)
                            sms_sent_this_upload = True
                            if alerted:
                                msg += f" | SMS sent → {', '.join(alerted)}"
                        else:
                            msg += " | (SMS already sent for this upload)"
                        stat_ph.error(msg)

            vid_ph.image(cv2.resize(rgb, (640, 360)), use_container_width=True)
            frame_n += 1
            time.sleep(delay)

        cap.release()
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

# ═══════════════════════════════════════════════
# IMAGE DETECTION
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("🖼 Image Detection")
    img_upload = st.file_uploader("Upload image…", type=["jpg", "jpeg", "png"])
    if img_upload:
        arr = np.asarray(bytearray(img_upload.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        _, texts, boxes = yolo.predictions(img)
        fw  = img.shape[1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        alerted_once = False
        for text, (x, y, w, h) in zip(texts, boxes):
            try:
                label, conf_str = text.split(" : ")
            except ValueError:
                continue
            lc  = label.strip().lower()
            rgb = draw_label(rgb, [lc, conf_str.strip()], x, y, fw)
            if lc == "emergency" and not alerted_once:
                d     = calculate_distance(w)
                names = alert_nearest(d, alert_radius, False, "image")
                st.success(
                    f"🚨 Emergency vehicle detected  |  ~{d:.0f}m away  |  "
                    f"SMS → {', '.join(names) if names else 'none (cooldown or no locations)'}"
                )
                alerted_once = True

        st.image(rgb, use_container_width=True)

# ═══════════════════════════════════════════════
# Alert Log
# ═══════════════════════════════════════════════
if st.session_state.alert_log:
    st.divider()
    st.subheader("📋 Alert History (this session)")
    for e in reversed(st.session_state.alert_log[-15:]):
        icon = "🔊" if e["audio"] else "👁"
        st.write(
            f"`{e['time']}` {icon} **{e['source']}**  |  "
            f"{e['dist']} m away  |  "
            f"Alerted: {', '.join(e['alerted']) if e['alerted'] else 'None'}"
        )
