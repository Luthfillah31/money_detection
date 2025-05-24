import streamlit as st
from ultralytics import YOLO
import cv2
import time
import os
import base64
from PIL import Image
import numpy as np
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av # For handling video frames with streamlit-webrtc
import threading # For thread-local data if needed, though session_state is often better

# --- Configuration ---
MODEL_PATH = "model/best.pt"  # Please ensure this path is correct
AUDIO_DIR = "."  # Directory where your .mp3 files are located
DETECTION_THRESHOLD_FOR_PLAY = 2.0  # Seconds an object must be detected before playing audio
DETECTION_COOLDOWN_AFTER_PLAY = 5.0  # Seconds to wait before playing audio for the same class again
# WEBCAM_ID is no longer directly used by cv2.VideoCapture for browser webcam
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

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
    except ImportError:
        success_message_placeholder.empty()
        st.error("Error loading YOLO model: The 'ultralytics' library might not be installed.")
    except Exception as e:
        success_message_placeholder.empty()
        st.error(f"An unexpected error occurred while loading YOLO model: {e}")
    return None

# --- Autoplay Audio Function ---
# This function will be called from the main Streamlit thread
def autoplay_audio_html(file_path: str, audio_placeholder):
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
        audio_placeholder.markdown(md, unsafe_allow_html=True)
        print(f"Attempted to play audio for: {file_path}")
    except Exception as e:
        print(f"Error embedding or playing audio {file_path}: {e}")

# --- Load Model ---
model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error("Application cannot start as the YOLO model failed to load.")
    st.stop()

# --- WebRTC Video Processor ---
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model # Use the globally loaded model
        # self.confidence_threshold will be updated from session_state
        # self.class_detection_state will be accessed from session_state
        # self.audio_to_play will be set in session_state

    def _initialize_session_state_if_needed(self):
        # Initialize session state keys if they don't exist
        # This helps manage state across reruns and interactions
        if 'class_detection_state' not in st.session_state:
            st.session_state.class_detection_state = {}
        if 'confidence_threshold_webrtc' not in st.session_state:
            st.session_state.confidence_threshold_webrtc = DEFAULT_CONFIDENCE_THRESHOLD
        if 'audio_to_play' not in st.session_state:
            st.session_state.audio_to_play = None
        if 'last_played_globally_time' not in st.session_state: # To avoid overlapping sounds if multiple classes detected quickly
            st.session_state.last_played_globally_time = 0


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._initialize_session_state_if_needed() # Ensure state is initialized

        img_bgr = frame.to_ndarray(format="bgr24")
        annotated_frame_bgr = img_bgr.copy()
        current_time = time.time()

        # Perform inference with the selected confidence threshold from session_state
        results = self.model(img_bgr, conf=st.session_state.confidence_threshold_webrtc, verbose=False)
        
        detected_classes_in_current_frame = set()

        if results and results[0].boxes:
            result = results[0]
            annotated_frame_bgr = result.plot(img=annotated_frame_bgr, conf=True)
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
                    time_since_last_played_for_class = current_time - class_state['last_played_time']
                    time_since_last_played_globally = current_time - st.session_state.get('last_played_globally_time', 0)


                    if (time_since_first_detection >= DETECTION_THRESHOLD_FOR_PLAY and
                        time_since_last_played_for_class >= DETECTION_COOLDOWN_AFTER_PLAY and
                        time_since_last_played_globally >= 1.0): # Add small global cooldown to prevent sound overlap

                        # Signal main thread to play audio
                        st.session_state.audio_to_play = class_name 
                        class_state['last_played_time'] = current_time
                        st.session_state.last_played_globally_time = current_time # Update global play time
                        # The main thread will pick this up and call autoplay_audio_html

                except Exception as e:
                    print(f"Error processing a detection box or audio logic in WebRTC: {e}")
        
        # Reset 'first_detected_time' for classes no longer in view
        for class_name_in_state in list(st.session_state.class_detection_state.keys()):
            if class_name_in_state not in detected_classes_in_current_frame:
                st.session_state.class_detection_state[class_name_in_state]['first_detected_time'] = current_time
        
        return av.VideoFrame.from_ndarray(annotated_frame_bgr, format="bgr24")

# --- Streamlit App Layout ---
st.title("Money Detection using YOLO")

# Initialize session state variables if they don't exist
if 'running' not in st.session_state:
    st.session_state.running = False
if 'uploaded_image_processed' not in st.session_state:
    st.session_state.uploaded_image_processed = False
if 'class_detection_state' not in st.session_state:
    st.session_state.class_detection_state = {}
if 'audio_to_play' not in st.session_state:
    st.session_state.audio_to_play = None
if 'confidence_threshold_webrtc' not in st.session_state: # For WebRTC
    st.session_state.confidence_threshold_webrtc = DEFAULT_CONFIDENCE_THRESHOLD
if 'confidence_threshold_upload' not in st.session_state: # For Upload
    st.session_state.confidence_threshold_upload = DEFAULT_CONFIDENCE_THRESHOLD


# --- Input Choice ---
input_choice = st.radio("Choose input source:", ("Webcam", "Upload Image"))

# --- Placeholders ---
audio_output_placeholder = st.empty() # For embedding audio
st_frame_placeholder = st.empty() # For displaying uploaded images or messages

# --- Main Processing Logic ---
if input_choice == "Webcam":
    st.session_state.uploaded_image_processed = False # Reset upload state
    st_frame_placeholder.empty() # Clear any previous image

    confidence_webcam_slider = st.slider(
        "Confidence Threshold (Webcam)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.confidence_threshold_webrtc, # Use specific session state
        step=0.01,
        help="Adjust for webcam detection.",
        key="webcam_conf_slider" # Unique key
    )
    # Update session state when slider changes
    st.session_state.confidence_threshold_webrtc = confidence_webcam_slider


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam Detection", key="start_webcam"):
            st.session_state.running = True
            st.session_state.class_detection_state = {} # Reset on start
            st.session_state.audio_to_play = None
            audio_output_placeholder.empty()
    with col2:
        if st.button("Stop Webcam Detection", key="stop_webcam"):
            st.session_state.running = False
            st.session_state.audio_to_play = None # Clear any pending audio
            audio_output_placeholder.empty()
            # st.session_state.class_detection_state = {} # Optionally reset state on stop

    if st.session_state.running:
        st.info("Webcam detection starting... Allow webcam access in your browser.")
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(
            key="yolo-webcam",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True, # Important for performance
        )
        if not webrtc_ctx.state.playing:
            st.session_state.running = False # If user stops it from webrtc UI or it fails
            audio_output_placeholder.empty()

    elif not st.session_state.running:
        st.info("Webcam detection is stopped.")
        audio_output_placeholder.empty()


elif input_choice == "Upload Image":
    st.session_state.running = False # Stop webcam if it was running
    audio_output_placeholder.empty() # Clear audio from webcam

    confidence_upload_slider = st.slider(
        "Confidence Threshold (Image Upload)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.confidence_threshold_upload, # Use specific session state
        step=0.01,
        help="Adjust for image upload detection.",
        key="upload_conf_slider" # Unique key
    )
    # Update session state when slider changes
    st.session_state.confidence_threshold_upload = confidence_upload_slider

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    # Separate start button for image processing
    if st.button("Process Uploaded Image", key="start_image_processing"):
        if uploaded_file is not None:
            st.session_state.uploaded_image_processed = False # Reset before processing
            st_frame_placeholder.empty()
            audio_output_placeholder.empty()
            st.session_state.class_detection_state = {} # Reset detection state for the new image


            st.write("Processing uploaded image...")
            try:
                image_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                image_np = np.array(pil_image)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                current_time = time.time()
                results = model(frame, conf=st.session_state.confidence_threshold_upload, verbose=False)
                annotated_frame = frame.copy()
                
                # Using a temporary set for audio to play for this image, to avoid re-playing for same class in one image
                played_audio_for_image = set()

                if results and results[0].boxes:
                    result = results[0]
                    annotated_frame = result.plot(img=annotated_frame, conf=True)
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        try:
                            class_id = int(boxes.cls[i].item())
                            class_name = model.names[class_id]

                            if class_name not in played_audio_for_image: # Play once per class per image
                                audio_file_name = f"{class_name}.mp3"
                                audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                                if os.path.exists(audio_file_path):
                                    # For multiple sounds in an image, they might overlap if played rapidly
                                    # We use one placeholder, so last sound "wins" or they are queued by browser if lucky
                                    autoplay_audio_html(audio_file_path, audio_output_placeholder)
                                    played_audio_for_image.add(class_name)
                                    time.sleep(0.5) # Small delay to allow audio to start / avoid too rapid triggering
                                else:
                                    print(f"Audio file not found for '{class_name}': {audio_file_path}")
                        except Exception as e:
                            print(f"Error processing a detection box or audio for uploaded image: {e}")
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Processed Image", use_container_width=True)
                st.session_state.uploaded_image_processed = True

            except Exception as e:
                st.error(f"An error occurred during image processing: {e}")
            # No finally block to clear audio here, as it should play after processing.
            # It will be cleared if user switches mode or processes new image.

        elif uploaded_file is None:
            st.warning("Please upload an image first before clicking 'Process Uploaded Image'.")
    
    if not st.session_state.uploaded_image_processed and uploaded_file is None:
         st_frame_placeholder.info("Upload an image and click 'Process Uploaded Image'.")


# --- Handle Audio Playback for WebRTC (Main Thread) ---
# This block runs in the main Streamlit thread and checks if the processor signaled to play audio
if st.session_state.get('audio_to_play'):
    class_to_play = st.session_state.audio_to_play
    audio_file_name = f"{class_to_play}.mp3"
    audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
    
    # Clear previous audio before playing new one
    # This ensures only one sound from WebRTC is attempted at a time.
    audio_output_placeholder.empty() 
    autoplay_audio_html(audio_file_path, audio_output_placeholder)
    
    st.session_state.audio_to_play = None # Reset the flag
