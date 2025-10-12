import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import threading
import time

# Streamlit config
st.set_page_config(page_title="Gest2Speak Demo", layout="centered")
st.title("🖐️ Gest2Speak: Gesture to Speech Demo")
st.markdown("Live webcam demo for gesture recognition")

# Settings
MODEL_PATH = "final_model.keras"
LABELS_PATH = "class_indices.json"
ROI_COORDS = (100, 100, 400, 400)

# Load model and label
@st.cache_resource
def load_model_and_labels():
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    labels = {v: k for k, v in labels.items()}  # reverse mapping
    return model, labels

model, labels = load_model_and_labels()

# TTS function that runs in a separate thread
def speak_text(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        pass  # Ignore TTS errors

# Session state for previous label
if 'prev_label' not in st.session_state:
    st.session_state.prev_label = None

# Camera orientation controls
st.sidebar.header("🎥 Camera Fix")
orientation = st.sidebar.selectbox(
    "Choose camera orientation:",
    ["Normal", "Rotate 90° CW", "Rotate 180°", "Rotate 270° CW", "Flip Vertical", "Flip Horizontal", "Flip Both"]
)

run_webcam = st.checkbox("Start Webcam")
mute_speech = st.checkbox("Mute Speech")

def fix_camera_orientation(frame, orientation):
    """Apply camera orientation fixes"""
    if orientation == "Normal":
        return frame
    elif orientation == "Rotate 90° CW":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == "Rotate 180°":
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif orientation == "Rotate 270° CW":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == "Flip Vertical":
        return cv2.flip(frame, 0)
    elif orientation == "Flip Horizontal":
        return cv2.flip(frame, 1)
    elif orientation == "Flip Both":
        return cv2.flip(frame, -1)  # Same as rotate 180°
    return frame

if run_webcam:
    # Create placeholders
    frame_placeholder = st.empty()
    prediction_text = st.empty()
    fps_text = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera.")
    else:
        prev_time = time.time()
        
        # Use a button to control the loop instead of while
        stop_button = st.button("Stop Webcam")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Cannot read from webcam")
                break

            # Fix camera orientation FIRST
            frame = fix_camera_orientation(frame, orientation)

            # Draw ROI box
            x1, y1, x2, y2 = ROI_COORDS
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = frame[y1:y2, x1:x2]

            # Preprocess for model
            if roi.size > 0:
                img = cv2.resize(roi, (224, 224))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                # Predict gesture
                pred = model.predict(img, verbose=0)
                pred_class = np.argmax(pred)
                confidence = np.max(pred)
                gesture_name = labels[pred_class]

                # Display prediction
                prediction_text.text(f"Predicted: {gesture_name} (Confidence: {confidence:.2f})")

                # Speak only if gesture changes, confidence is high, and not muted
                if (gesture_name != st.session_state.prev_label and 
                    confidence > 0.7 and not mute_speech):
                    # Run TTS in separate thread to avoid blocking
                    tts_thread = threading.Thread(target=speak_text, args=(gesture_name,))
                    tts_thread.daemon = True
                    tts_thread.start()
                    st.session_state.prev_label = gesture_name

                # Overlay gesture name on frame
                cv2.putText(frame, f"{gesture_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate and show FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_text.text(f"FPS: {fps:.1f}")

            # Show frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Small delay to prevent overwhelming the UI
            time.sleep(0.1)

        cap.release()
        st.success("🛑 Webcam stopped.")

else:
    st.info("👆 Check 'Start Webcam' to begin gesture recognition")
    st.info("📝 Use the sidebar dropdown to fix camera orientation if needed")

# Alternative: Image upload option
st.subheader("📷 Test with Image Upload")
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", channels="BGR")
    
    with col2:
        # Preprocess image
        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized.astype("float32") / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        pred = model.predict(img_batch, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        gesture_name = labels[pred_class]
        
        st.write(f"**Prediction:** {gesture_name}")
        st.write(f"**Confidence:** {confidence:.2f}")
