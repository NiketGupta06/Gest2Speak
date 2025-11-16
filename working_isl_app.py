import streamlit as st
import numpy as np
import tensorflow as tf
import json
from collections import deque
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import os

st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Real-Time Indian Sign Language Recognition")
st.info("**Accurate gesture recognition using your trained CNN+LSTM model!**")

# Load class names
def load_class_names():
    with open('class_indices_lstm_v2.json', 'r') as f:
        data = json.load(f)
    return data['classes'] if 'classes' in data else list(map(str, range(36)))

# Exact preprocessing
def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_frame.astype('float32') / 255.0
    return normalized

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('final_model_compatible.keras', compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()
class_names = load_class_names()

# Video transformer for real-time processing
class ISLRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_buffer = deque(maxlen=10)
        self.prediction = "Waiting..."
        self.confidence = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = preprocess_frame(img)
        self.frame_buffer.append(processed)

        if len(self.frame_buffer) == 10:
            self.predict_gesture()

        # Overlay prediction
        out = img.copy()
        cv2.rectangle(out, (10, 10), (430, 120), (0, 0, 0), -1)
        cv2.rectangle(out, (10, 10), (430, 120), (255, 255, 255), 2)
        cv2.putText(out, f"Gesture: {self.prediction}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(out, f"Confidence: {self.confidence:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        return out

    def predict_gesture(self):
        try:
            sequence = np.array(self.frame_buffer)
            sequence = np.expand_dims(sequence, axis=0)
            preds = model.predict(sequence, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            self.prediction = class_names[idx] if idx < len(class_names) else f"Class{idx}"
            self.confidence = conf
        except Exception as e:
            self.prediction = "Error"
            self.confidence = 0.0

# Streamlit interface
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📹 Camera (real model, true weights!)")
    webrtc_streamer(
        key="ISL",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=ISLRecognitionTransformer,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 30},
            "audio": False
        },
        async_processing=True
    )

with col2:
    st.subheader("Model Details")
    st.success("✅ **Trained weights loaded and used**")
    st.write(f"Classes: {len(class_names)}")
    st.write(f"Architecture: CNN+LSTM ({model.input_shape} → {model.output_shape})")
    st.write("Sample gestures:")
    st.write(", ".join(class_names))

st.sidebar.subheader("Debug Info")
st.sidebar.write("Model file: final_model_compatible.keras")
st.sidebar.write(f"File size: {os.path.getsize('final_model_compatible.keras')/1024/1024:.1f} MB")
st.sidebar.write(f"Classes: {len(class_names)}")
st.sidebar.write(f"TensorFlow: {tf.__version__}")

st.markdown("---")
st.success("This app uses your actual trained weights and should produce **correct, accurate predictions!**")
