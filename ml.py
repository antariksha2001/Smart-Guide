import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
from io import BytesIO
import base64
import speech_recognition as sr
import threading
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # For WebRTC
import easyocr  # Import EasyOCR for text recognition
import tempfile

# Initialize the OCR reader
reader = easyocr.Reader(['en', 'hi'])  # You can add more languages

# Load the YOLO model
net = cv2.dnn.readNet('https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?select=yolov3.weights', 'https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?select=yolov3.cfg')
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Define the classes for YOLO
with open('https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data?select=coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Function to calculate distance (dummy function, modify according to camera specifics)
def calculate_distance(width, known_width=30, focal_length=700):
    distance = (known_width * focal_length) / width
    return distance

# Function to detect objects
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = []
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = calculate_distance(w)  # Calculate distance based on the width of the object
            results.append({
                "label": label,
                "confidence": confidences[i],
                "box": (x, y, w, h),
                "distance": distance
            })

    return results

# Function to detect text using OCR
def detect_text(frame, language):
    # Convert the frame to grayscale (optional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the image
    result = reader.readtext(gray)

    if len(result) > 0:
        # Loop through detected text
        for (bbox, text, prob) in result:
            if prob > 0.5:  # Confidence threshold
                # Draw bounding box around text
                cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Speak the detected text
                speak(text, lang=language[1])

    return frame

# Function to handle frame processing
def process_frame(frame, language):
    # Object detection
    detections = detect_objects(frame)
    
    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["label"]
        distance = detection["distance"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label} ({distance:.2f} cm)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Speak object detection result and distance
        speak(f'{label} detected at {distance:.2f} centimeters', lang=language[1])

    # OCR detection
    frame = detect_text(frame, language)

    return frame

# Function to speak using gTTS and play sound using HTML audio
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    audio_base64 = base64.b64encode(audio_data.read()).decode()
    
    # Display the audio element to play sound on mobile
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Streamlit design
def main():
    st.set_page_config(page_title="Smart Guide with Text Detection", layout="wide")

    # Add a background image
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_base64 = get_base64_image("ojt/ist.jpg")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            font-family: 'Arial', sans-serif;
        }}
        .card-box {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
            max-width: 700px;
            margin: 50px auto;
            text-align: center;
        }}
        .stButton > button {{
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            background-color: #007bff;
            color: white;
            border: none;
            font-size: 18px;
        }}
        .stButton > button:hover {{
            background-color: #0056b3;
        }}
        h1 {{
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #555;
            font-size: 2rem;
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Smart Guide with Text Detection")

    st.markdown(
        """
        <div class="card-box">
            <h2>Experience Object and Text Detection in Real-Time</h2>
            <p>Select your preferred language and video source to begin. The Smart Guide will assist you by detecting objects and reading text.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    language = st.selectbox(
        "Select Language", 
        [
            ("English", "en"), 
            ("Bengali", "bn"), 
            ("Hindi", "hi"),
            ("Tamil", "ta"),
            ("Telugu", "te"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
            ("Korean", "ko")
        ]
    )

    video_source = st.selectbox("Select Video Source", ["Webcam", "Upload Video", "Upload Image"])

    if video_source == "Webcam":
        webrtc_streamer(key="object_detection", video_transformer_factory=VideoTransformerBase)

    if video_source == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_file.read())
                cap = cv2.VideoCapture(temp_video.name)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process the frame
                    frame = process_frame(frame, language)

                    # Display the frame
                    st.image(frame, channels="BGR")

                cap.release()

    if video_source == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Process the image
            frame = process_frame(image, language)

            # Display the image
            st.image(frame, channels="BGR")

if __name__ == "__main__":
    main()
