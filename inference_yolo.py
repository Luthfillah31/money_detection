import streamlit as st
from ultralytics import YOLO
import cv2
import time
import os
import base64
from PIL import Image
import numpy as np
import io
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration, WebRtcMode

# --- Configuration ---
MODEL_PATH = "model/best.pt"  # Please ensure this path is correct
AUDIO_DIR = "."  # Directory where your .mp3 files are located
DETECTION_THRESHOLD_FOR_PLAY = 2.0  # Seconds an object must be detected before playing audio
DETECTION_COOLDOWN_AFTER_PLAY = 5.0  # Seconds to wait before playing audio for the same class again
# WEBCAM_ID is no longer used for streamlit-webrtc
DEFAULT_CONFIDENCE_THRESHOLD = 0.25  # Default confidence threshold for YOLO model

# --- YOLO Model Loading (Cache to load only once) ---
@st.cache_resource
def load_yolo_model(path):
    success_message_placeholder = st.empty()
    model = None
    try:
        model = YOLO(path)
        success_message_placeholder.success(
            f"YOLO model loaded successfully.\nModel path: {path}"
        )
        time.sleep(3)
        success_message_placeholder.empty()
        return model
    except FileNotFoundError:
        success_message_placeholder.empty()
        st.error(f"Error loading YOLO model: Model file not found at {path}")
        st.error("Please ensure the model path is correct and the file exists.")
        return None
    except ImportError:
        success_message_placeholder.empty()
        st.error("Error loading YOLO model: The 'ultralytics' library or YOLO class might not be installed or imported correctly.")
        st.error("Please install it (e.g., pip install ultralytics) and ensure it's imported.")
        return None
    except Exception as e:
        success_message_placeholder.empty()
        st.error(f"An unexpected error occurred while loading YOLO model from {path}: {e}")
        st.error("Please check the model file and dependencies.")
        return None

# --- Autoplay Audio Function ---
def autoplay_audio_html(file_path: str, audio_placeholder_to_use):
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
            <audio controls autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
            </audio>
            """
            audio_placeholder_to_use.markdown(md, unsafe_allow_html=True)
            print(f"Attempted to play audio for: {file_path}")
    except Exception as e:
        print(f"Error embedding or playing audio {file_path}: {e}")

# --- Load Model ---
model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error("Application cannot start as the YOLO model failed to load. Please check the path and model file.")
    st.stop()

# --- Streamlit App Layout ---
st.title("Money Detection using YOLO")

# --- Input Choice ---
input_choice = st.radio("Choose input source:", ("Webcam (via your browser)", "Upload Image"))

# --- Confidence Threshold Slider ---
# Ensure the key is unique and accessible
confidence_slider = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.01,
    help="Adjust the minimum confidence for object detection.",
    key="confidence_slider_value" # Added a key
)

# --- Session State Initialization ---
if 'class_detection_state' not in st.session_state:
    st.session_state.class_detection_state = {}
if 'audio_trigger' not in st.session_state: # To trigger audio from the main thread
    st.session_state.audio_trigger = None
if 'last_audio_trigger_processed' not in st.session_state:
    st.session_state.last_audio_trigger_processed = None
if 'running_image_processing' not in st.session_state: # For image upload start/stop
    st.session_state.running_image_processing = False
if 'uploaded_image_processed' not in st.session_state:
    st.session_state.uploaded_image_processed = False


# --- RTC Configuration for WebRTC ---
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# --- Placeholder for audio controlled by the main script ---
main_audio_placeholder = st.empty()


# --- Video Transformer for WebRTC ---
class YoloVideoProcessor(VideoProcessorBase): # Renamed class and changed base class
    def __init__(self):
        self.model = model
        # self.confidence_threshold = st.session_state.confidence_slider_value # Access directly in recv

    def _process_frame_for_yolo_and_audio(self, frame: np.ndarray):
        # ... (this internal logic remains the same) ...
        current_time = time.time()
        current_confidence = st.session_state.confidence_slider_value
        results = self.model(frame, conf=current_confidence, verbose=False)
        annotated_frame = frame.copy()
        detected_classes_in_current_frame = set()

        if results and results[0].boxes:
            result = results[0]
            annotated_frame = result.plot(img=annotated_frame, conf=True)
            boxes = result.boxes

            for i in range(len(boxes)):
                try:
                    class_id = int(boxes.cls[i].item())
                    class_name = self.model.names[class_id]
                    detected_classes_in_current_frame.add(class_name)

                    if class_name not in st.session_state.class_detection_state:
                        st.session_state.class_detection_state[class_name] = {
                            'first_detected_time': current_time,
                            'last_played_time': 0
                        }
                    
                    class_state = st.session_state.class_detection_state[class_name]
                    time_since_first_detection = current_time - class_state['first_detected_time']
                    time_since_last_played = current_time - class_state['last_played_time']

                    if (time_since_first_detection >= DETECTION_THRESHOLD_FOR_PLAY and
                        time_since_last_played >= DETECTION_COOLDOWN_AFTER_PLAY):
                        st.session_state.audio_trigger = {
                            "class_name": class_name, 
                            "eligible_time": current_time,
                            "id": f"{class_name}_{current_time}"
                        }
                except Exception as e:
                    print(f"Error processing a detection box or audio logic in processor: {e}")
        
        for class_name_in_state in list(st.session_state.class_detection_state.keys()):
            if class_name_in_state not in detected_classes_in_current_frame:
                st.session_state.class_detection_state[class_name_in_state]['first_detected_time'] = current_time
        
        return annotated_frame

    # The recv method signature usually expects an av.VideoFrame
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated_img = self._process_frame_for_yolo_and_audio(img)
        # Ensure the returned frame is also an av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# ... (in the main app logic for webcam) ...
if input_choice == "Webcam (via your browser)":
    st.info("Webcam will start via your browser. Grant permissions when prompted. Detection is continuous.")
    st.session_state.running_image_processing = False

    webrtc_ctx = webrtc_streamer(
        key="yolo-object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YoloVideoProcessor, # Changed argument name and class name
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        # You can add messages or UI elements that are specific to when the webcam is active
        pass
    else:
        # Optionally, clear states when webcam stops if desired
        # st.session_state.class_detection_state = {} # This might be too aggressive
        # st.session_state.audio_trigger = None
        pass


elif input_choice == "Upload Image":
    # Control Buttons for Image Upload
    col1, col2 = st.columns(2)
    with col1:
        start_button_img = st.button("Start Detection on Image", key="start_image_detection")
    with col2:
        stop_button_img = st.button("Stop/Clear Image Detection", key="stop_image_detection")

    if start_button_img:
        st.session_state.running_image_processing = True
        st.session_state.uploaded_image_processed = False # Reset for new detection
    if stop_button_img:
        st.session_state.running_image_processing = False
        st.session_state.uploaded_image_processed = True # Mark as "stopped"
        # Clear placeholders and state
        st_frame_placeholder_img.empty()
        main_audio_placeholder.empty()
        st.session_state.class_detection_state = {}
        st.session_state.audio_trigger = None
        st.info("Image detection stopped and results cleared.")


    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader_img")
    st_frame_placeholder_img = st.empty() # Placeholder for uploaded image display

    if st.session_state.running_image_processing and uploaded_file is not None and not st.session_state.uploaded_image_processed:
        st.write("Processing uploaded image...")
        try:
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            current_time = time.time()
            current_confidence = st.session_state.confidence_slider_value # Use session state for confidence
            results = model(frame, conf=current_confidence, verbose=False)
            annotated_frame = frame.copy()
            detected_classes_in_current_frame = set()

            if results and results[0].boxes:
                result = results[0]
                annotated_frame = result.plot(img=annotated_frame, conf=True)
                boxes = result.boxes

                for i in range(len(boxes)):
                    try:
                        class_id = int(boxes.cls[i].item())
                        class_name = model.names[class_id]
                        detected_classes_in_current_frame.add(class_name)
                        
                        # For image upload, audio logic is simpler: play if detected and cooldown allows
                        if class_name not in st.session_state.class_detection_state:
                            st.session_state.class_detection_state[class_name] = {
                                'first_detected_time': current_time, # Less critical for single image
                                'last_played_time': 0
                            }
                        
                        class_state = st.session_state.class_detection_state[class_name]
                        time_since_last_played = current_time - class_state['last_played_time']

                        # For image, we don't need the 'DETECTION_THRESHOLD_FOR_PLAY' strictly,
                        # just the cooldown. Play immediately if detected.
                        if time_since_last_played >= DETECTION_COOLDOWN_AFTER_PLAY:
                            st.session_state.audio_trigger = {
                                "class_name": class_name,
                                "eligible_time": current_time,
                                "id": f"{class_name}_{current_time}_img" # Unique ID
                            }
                            # The main loop will handle playback and updating 'last_played_time'
                    except Exception as e:
                        print(f"Error processing a detection box for uploaded image: {e}")

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame_placeholder_img.image(annotated_frame_rgb, channels="RGB", caption="Processed Image", use_container_width=True)
            
            st.session_state.uploaded_image_processed = True # Mark as processed
            # st.session_state.running_image_processing = False # Optionally stop after one image

        except Exception as e:
            st.error(f"An error occurred during image processing: {e}")
            st.session_state.running_image_processing = False
    
    elif not st.session_state.running_image_processing and not st.session_state.uploaded_image_processed:
        st_frame_placeholder_img.empty()
        st.info("Upload an image and click 'Start Detection on Image' to process it.")
    elif st.session_state.uploaded_image_processed and not st.session_state.running_image_processing:
         st.info("Image processed or detection stopped. Upload a new one or click 'Start Detection on Image' again.")


# --- Audio Playback Logic (handles triggers from WebRTC or Image Upload) ---
if st.session_state.audio_trigger and \
   st.session_state.audio_trigger.get("id") != st.session_state.last_audio_trigger_processed:
    
    audio_info = st.session_state.audio_trigger
    class_name_to_play = audio_info['class_name']
    eligible_time = audio_info['eligible_time']

    # Ensure class_name exists in state (should be, but good check)
    if class_name_to_play in st.session_state.class_detection_state:
        audio_file_name = f"{class_name_to_play}.mp3"
        audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
        
        # Clear previous audio from the main placeholder and play new
        main_audio_placeholder.empty() 
        autoplay_audio_html(audio_file_path, main_audio_placeholder)
        
        # Update the last_played_time for this class
        st.session_state.class_detection_state[class_name_to_play]['last_played_time'] = eligible_time
        st.session_state.last_audio_trigger_processed = audio_info.get("id")
        
        # Optional: Reset trigger after processing if it's a one-shot, 
        # but using last_audio_trigger_processed ID check is more robust.
        # st.session_state.audio_trigger = None 
    else:
        print(f"Warning: Class {class_name_to_play} not found in detection state for audio trigger.")
