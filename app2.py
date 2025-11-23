import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# robust import for load_model to accommodate different environments / linters
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        from keras.models import load_model
    except Exception:
        load_model = tf.keras.models.load_model

import json
import time

# Streamlit config
st.set_page_config(
    page_title="Gest2Speak - Deep Learning Gesture Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS optimized for dark mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        padding: 0;
        background: #0a0e27;
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .content-wrapper {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b4e 100%);
        padding: 4rem 3rem;
        border-radius: 25px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        border: 1px solid rgba(139, 92, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at top right, rgba(139, 92, 246, 0.1), transparent 50%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    .hero-title {
        background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        color: #e0e7ff;
        font-size: 1.5rem;
        margin-top: 1rem;
        font-weight: 600;
        line-height: 1.5;
    }
    
    .hero-description {
        color: #c7d2fe;
        font-size: 1.1rem;
        margin-top: 1.5rem;
        max-width: 1000px;
        line-height: 1.8;
    }
    
    .section-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(139, 92, 246, 0.3);
    }
    
    .section-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .section-title {
        color: #e0e7ff;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .section-content {
        color: #c7d2fe;
        font-size: 1.05rem;
        line-height: 1.8;
        text-align: justify;
    }
    
    .objectives-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .objective-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(139, 92, 246, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .objective-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: 0;
    }
    
    .objective-card:hover::before {
        opacity: 0.15;
    }
    
    .objective-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(139, 92, 246, 0.4);
        border-color: #8b5cf6;
    }
    
    .objective-content {
        position: relative;
        z-index: 1;
    }
    
    .objective-number {
        background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1rem;
        letter-spacing: -2px;
    }
    
    .objective-text {
        color: #e0e7ff;
        font-size: 1.05rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .tech-badge {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        color: #c4b5fd;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 2px solid rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
    }
    
    .tech-badge:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
        border-color: #8b5cf6;
    }
    
    .control-panel {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .control-title {
        color: #e0e7ff;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
        height: 100%;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-box:hover {
        transform: scale(1.02);
    }
    
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 900;
        margin-top: 0.75rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .webcam-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .webcam-title {
        color: #e0e7ff;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .instructions-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(245, 158, 11, 0.15) 100%);
        border-left: 5px solid #f59e0b;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.2);
    }
    
    .instructions-title {
        color: #fde68a;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    }
    
    .instruction-item {
        color: #fef3c7;
        font-size: 1.05rem;
        margin: 0.75rem 0;
        padding-left: 2rem;
        position: relative;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .instruction-item::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #fbbf24;
        font-weight: 900;
        font-size: 1.2rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: #6ee7b7;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: #c7d2fe;
        font-size: 1rem;
        margin-top: 4rem;
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(49, 46, 129, 0.6) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .footer-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #e0e7ff;
    }
    
    .footer-text {
        opacity: 0.9;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .stFileUploader {
        background: rgba(139, 92, 246, 0.1);
        border-radius: 15px;
        padding: 2rem;
        border: 2px dashed #8b5cf6;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        height: 8px;
        border-radius: 10px;
    }
    
    hr {
        margin: 3rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #e0e7ff;
    }
    
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Content wrapper
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-badge">🏆 Deep Learning Project</div>
        <h1 class="hero-title">🖐️ Gest2Speak</h1>
        <p class="hero-subtitle">Deep Learning-Based Gesture-to-Speech Conversion System</p>
        <p class="hero-description">
            A real-world solution empowering individuals with speech impairments to communicate 
            using intuitive hand gestures, powered by CNN + LSTM.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Problem Description Section
st.markdown("""
<div class="section-container">
    <div class="section-header">
        <span class="section-icon">🎯</span>
        <h2 class="section-title">Problem Statement</h2>
    </div>
    <p class="section-content">
        Individuals with speech impairments or non-verbal conditions often face significant challenges 
        in communicating effectively with others. Traditional sign language requires interpreters or 
        prior knowledge by the listener, creating barriers to seamless communication and limiting accessibility. 
        This project addresses these challenges by developing an intelligent gesture-to-speech conversion system 
        using advanced deep learning techniques—specifically CNN for spatial feature extraction and gesture 
        recognition, combined with LSTM for temporal sequence modeling.
    </p>
</div>
""", unsafe_allow_html=True)

# Objectives Section
st.markdown("""
<div class="section-container">
    <div class="section-header">
        <span class="section-icon">🎓</span>
        <h2 class="section-title">Research Objectives</h2>
    </div>
    <div class="objectives-grid">
        <div class="objective-card">
            <div class="objective-content">
                <div class="objective-number">01</div>
                <div class="objective-text">
                    Implement CNN + LSTM for robust gesture recognition in real-world scenarios.
                </div>
            </div>
        </div>
        <div class="objective-card">
            <div class="objective-content">
                <div class="objective-number">02</div>
                <div class="objective-text">
                    Work with real-time computer vision and video processing using OpenCV.
                </div>
            </div>
        </div>
        <div class="objective-card">
            <div class="objective-content">
                <div class="objective-number">03</div>
                <div class="objective-text">
                    Build an end-to-end gesture-to-speech pipeline with high prediction accuracy.
                </div>
            </div>
        </div>
        <div class="objective-card">
            <div class="objective-content">
                <div class="objective-number">04</div>
                <div class="objective-text">
                    Provide an accessible interface for users to interact using their webcam.
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Technology Stack Section
st.markdown("""
<div class="section-container">
    <div class="section-header">
        <span class="section-icon">💻</span>
        <h2 class="section-title">Technology Stack</h2>
    </div>
    <div class="tech-stack">
        <div class="tech-badge">🐍 Python 3.x</div>
        <div class="tech-badge">🎥 OpenCV</div>
        <div class="tech-badge">🧠 TensorFlow/Keras</div>
        <div class="tech-badge">📊 CNN Architecture</div>
        <div class="tech-badge">🔄 LSTM Networks</div>
        <div class="tech-badge">⚡ Streamlit</div>
        <div class="tech-badge">📸 Webcam Input</div>
        <div class="tech-badge">🤖 Deep Learning</div>
        <div class="tech-badge">🎨 Computer Vision</div>
        <div class="tech-badge">📐 NumPy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Settings
MODEL_PATH = "final_model.keras"
LABELS_PATH = "class_indices.json"
ROI_COORDS = (100, 100, 400, 400)  # x1, y1, x2, y2 on the captured frame

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
        labels = {v: k for k, v in labels.items()}  # index -> label
    return model, labels

with st.spinner('🔄 Loading CNN-LSTM Neural Network...'):
    model, labels = load_model_and_labels()
st.markdown('<div class="status-success">✅ Model loaded successfully. Ready for inference.</div>', unsafe_allow_html=True)

# Control Panel
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.markdown('<div class="control-title">🎛️ Live Demo Control Panel</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 2])
with col1:
    use_webcam = st.checkbox("🎥 Use Webcam (Snapshot)", help="Capture a frame from your webcam and run the model.")
with col2:
    show_instructions = st.checkbox("💡 Show User Guide", value=True)

st.markdown('</div>', unsafe_allow_html=True)

# Instructions
if show_instructions:
    st.markdown("""
    <div class="instructions-box">
        <div class="instructions-title">💡 Quick Start Guide</div>
        <div class="instruction-item">Tick the webcam checkbox to show the camera capture widget.</div>
        <div class="instruction-item">Allow camera access when your browser asks for permission.</div>
        <div class="instruction-item">Place your hand inside the ROI box area and click "Take Photo".</div>
        <div class="instruction-item">The system will predict the gesture and show the result with confidence.</div>
    </div>
    """, unsafe_allow_html=True)

# ===========================
# Webcam snapshot + inference
# ===========================

if use_webcam:
    st.markdown("---")
    webcam_col, stats_col = st.columns([2.5, 1])

    with webcam_col:
        st.markdown('<div class="webcam-box">', unsafe_allow_html=True)
        st.markdown('<div class="webcam-title">📹 Webcam Capture</div>', unsafe_allow_html=True)
        camera_image = st.camera_input("Show your gesture and click 'Take Photo'")
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with stats_col:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown('<div class="webcam-title">📊 Prediction</div>', unsafe_allow_html=True)
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if camera_image is not None:
        # Convert webcam image (UploadedFile) to OpenCV BGR array
        file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)  # BGR

        # Optional: flip vertically+ horizontally to match your training orientation
        frame = cv2.flip(frame, -1)

        h, w, _ = frame.shape
        x1, y1, x2, y2 = ROI_COORDS
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (139, 92, 246), 5)
        label_text = "HAND REGION"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        top_y = max(0, y1 - 40)
        cv2.rectangle(
            frame,
            (x1, top_y),
            (x1 + label_size[0] + 20, y1),
            (139, 92, 246),
            -1
        )
        cv2.putText(
            frame,
            label_text,
            (x1 + 10, max(20, y1 - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        roi = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

        if roi is not None and roi.size > 0:
            img = cv2.resize(roi, (224, 224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)
            pred_class = int(np.argmax(pred))
            confidence = float(np.max(pred))
            gesture_name = labels.get(pred_class, str(pred_class))

            # Overlays on frame
            gesture_text = f"{gesture_name}"
            conf_text = f"Confidence: {confidence:.1%}"

            text_size = cv2.getTextSize(
                gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )[0]
            box_y1 = max(0, y1 - 100)
            box_y2 = max(0, y1 - 50)
            cv2.rectangle(
                frame,
                (x1, box_y1),
                (x1 + text_size[0] + 20, box_y2),
                (139, 92, 246),
                -1
            )
            cv2.putText(
                frame,
                gesture_text,
                (x1 + 10, max(30, y1 - 65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )

            conf_size = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            conf_box_y1 = min(h - 50, y2 + 10)
            conf_box_y2 = min(h - 10, y2 + 50)
            cv2.rectangle(
                frame,
                (x1, conf_box_y1),
                (x1 + conf_size[0] + 20, conf_box_y2),
                (139, 92, 246),
                -1
            )
            cv2.putText(
                frame,
                conf_text,
                (x1 + 10, conf_box_y2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Stats on right
            prediction_placeholder.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Detected Gesture</div>
                <div class="stat-value">{gesture_name}</div>
            </div>
            """, unsafe_allow_html=True)

            confidence_placeholder.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Confidence Score</div>
                <div class="stat-value">{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        # Show processed frame (BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

# =======================
# Static Image Testing
# =======================

st.markdown("---")
st.markdown("""
<div class="section-container">
    <div class="section-header">
        <span class="section-icon">📷</span>
        <h2 class="section-title">Static Image Testing</h2>
    </div>
    <p class="section-content">
        Upload a gesture image to evaluate the CNN-LSTM model's performance without using the webcam.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a gesture image file",
    type=['png', 'jpg', 'jpeg'],
    help="Supported formats: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("#### 📸 Original Image")
        st.image(image, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("#### 🔮 Model Prediction")

        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized.astype("float32") / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        with st.spinner('🧠 Processing image through CNN-LSTM model...'):
            pred = model.predict(img_batch, verbose=0)
            pred_class = int(np.argmax(pred))
            confidence = float(np.max(pred))
            gesture_name = labels.get(pred_class, str(pred_class))

        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Detected Gesture</div>
            <div class="stat-value">{gesture_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Confidence Score</div>
            <div class="stat-value">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-title">Gest2Speak</div>
    <p class="footer-text">Deep Learning-Based Gesture Recognition System</p>
    <p class="footer-text">Powered by CNN + LSTM 🧠 | Built with TensorFlow & Streamlit ⚡</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
