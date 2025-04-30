import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# ==== Setup ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load YOLO model once
model = YOLO("yolov8n-face.pt").to(device)
model.fuse()  # Optimize model for inference

st.title("🎯 Real-Time Face Detection with YOLOv8")

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ==== Processor ====
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.frame_count = 0
        self.last_result = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        if self.frame_count % 5 == 0:
            # Resize to smaller size for faster processing
            img_small = cv2.resize(img, (320, 240))

            # Run inference
            results = self.model(img_small, conf=0.3)
            self.last_result = results[0]

        if self.last_result:
            for box in self.last_result.boxes:
                if box.conf[0] < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Scale coords to match original frame size
                x1 = int(x1 * img.shape[1] / 320)
                x2 = int(x2 * img.shape[1] / 320)
                y1 = int(y1 * img.shape[0] / 240)
                y2 = int(y2 * img.shape[0] / 240)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Face {box.conf[0]:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==== Start Stream ====
webrtc_streamer(
    key="face-detection",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
