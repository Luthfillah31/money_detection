import streamlit as st
from ultralytics import YOLO
import cv2
import time
import os
import base64
from PIL import Image
import numpy as np
import io

# --- Configuration ---
MODEL_PATH = "model/best.pt"  # Please ensure this path is correct
AUDIO_DIR = "."  # Directory where your .mp3 files are located
# DETECTION_THRESHOLD_FOR_PLAY is less relevant for single camera snapshots
DETECTION_COOLDOWN_AFTER_PLAY = 5.0  # Seconds to wait before playing audio for the same class again
DEFAULT_CONFIDENCE_THRESHOLD = 0.25  # Default confidence threshold for YOLO model

# --- YOLO Model Loading (Cache to load only once) ---
@st.cache_resource
def load_yolo_model(path):
    """
    Loads the YOLO model and caches it.
    Success messages are displayed for a few seconds.
    Error messages will persist.
    """
    success_message_placeholder = st.empty()
    model = None
    try:
        model = YOLO(path)
        success_message_placeholder.success(
            f"YOLO model loaded successfully.\nModel path: {path}"
        )
        time.sleep(3)  # Keep success message for 3 seconds
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
def autoplay_audio_html(file_path: str, audio_placeholder):
    """
    Embeds an audio file as base64 into an HTML audio tag with autoplay.
    Updates the content of a given Streamlit placeholder.
    """
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
    st.error("Application cannot start as the YOLO model failed to load. Please check the path and model file.")
    st.stop()

# --- Streamlit App Layout ---
st.title("Money Detection using YOLO 💰")

# --- Input Choice ---
input_choice = st.radio("Choose input source:", ("Webcam", "Upload Image"))

# --- Confidence Threshold Slider ---
confidence_slider = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.01,
    help="Adjust the minimum confidence for object detection. Lower values detect more objects but may include more false positives."
)

# --- Control Buttons ---
# The 'running' state now primarily gates whether processing occurs after input.
if 'running' not in st.session_state:
    st.session_state.running = False
if 'uploaded_image_processed' not in st.session_state:
    st.session_state.uploaded_image_processed = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start/Process", key="start_processing"): # Changed label for clarity
        st.session_state.running = True
        st.session_state.uploaded_image_processed = False # Reset for new detection
with col2:
    if st.button("Stop/Clear", key="stop_clear"): # Changed label
        st.session_state.running = False
        st.session_state.uploaded_image_processed = False


# --- State for tracking detection times and played status ---
if 'class_detection_state' not in st.session_state:
    st.session_state.class_detection_state = {}

# --- Placeholders ---
audio_output_placeholder = st.empty()  # For embedding audio
st_frame_placeholder = st.empty()  # For displaying video frames or images

# --- Function to process a single frame (from webcam or upload) ---
def process_frame(frame_to_process, current_time):
    global model, confidence_slider, audio_output_placeholder # Added audio_output_placeholder
    results = model(frame_to_process, conf=confidence_slider, verbose=False)
    annotated_frame = frame_to_process.copy()
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

                if class_name not in st.session_state.class_detection_state:
                    st.session_state.class_detection_state[class_name] = {
                        'last_played_time': 0
                    }
                
                class_state = st.session_state.class_detection_state[class_name]
                time_since_last_played = current_time - class_state['last_played_time']

                # Simplified audio logic for single frames: Play if cooldown met
                if time_since_last_played >= DETECTION_COOLDOWN_AFTER_PLAY:
                    audio_file_name = f"{class_name}.mp3"
                    audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                    if os.path.exists(audio_file_path):
                        # For single frames, manage audio placeholder carefully if multiple sounds can play
                        # This might need a more sophisticated queue or stacking if sounds overlap
                        temp_audio_placeholder = st.empty() # Use temporary for each sound
                        autoplay_audio_html(audio_file_path, temp_audio_placeholder)
                        class_state['last_played_time'] = current_time
                        # time.sleep(0.5) # Optional small delay to allow audio to start
                    else:
                        print(f"Audio file not found for '{class_name}': {audio_file_path}")
            except Exception as e:
                print(f"Error processing a detection box or audio logic: {e}")
    
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)


# --- Main Processing Logic ---
try:
    if input_choice == "Webcam":
        st.info("Press 'Start/Process' then use the camera widget below to take a picture.")
        img_file_buffer = st.camera_input("Take a picture for detection", key="webcam_photo", disabled=not st.session_state.running)

        if img_file_buffer is not None and st.session_state.running:
            st.write("Processing webcam image...")
            try:
                pil_image = Image.open(img_file_buffer)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                
                image_np = np.array(pil_image)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                current_time = time.time()
                process_frame(frame, current_time)
                st.success("Webcam image processed!")
                # Consider if 'running' should be set to False after one pic, or allow multiple
                # For now, it allows multiple if user takes another picture while 'running' is true

            except Exception as e:
                st.error(f"An error occurred during webcam image processing: {e}")
                st.session_state.running = False # Stop on error
            finally:
                # Reset class detection state if desired after each webcam pic processing
                # st.session_state.class_detection_state = {} 
                pass


    elif input_choice == "Upload Image":
        st.info("Upload an image, then press 'Start/Process'.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

        if uploaded_file is not None and st.session_state.running and not st.session_state.uploaded_image_processed:
            st.write("Processing uploaded image...")
            try:
                image_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                
                image_np = np.array(pil_image)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                current_time = time.time()
                process_frame(frame, current_time)
                
                st.session_state.uploaded_image_processed = True
                st.success("Uploaded image processed!")
                # st.session_state.running = False # Stop detection automatically after processing one image

            except Exception as e:
                st.error(f"An error occurred during image processing: {e}")
                st.session_state.running = False
            finally:
                # Reset state after processing an image
                # st.session_state.class_detection_state = {} 
                pass
    
    # UI updates based on running state
    if not st.session_state.running:
        if not st.session_state.uploaded_image_processed and input_choice == "Upload Image":
             st.info("Upload an image and click 'Start/Process' to begin.")
        elif input_choice == "Webcam":
             st.info("Click 'Start/Process' to enable the camera widget.")
        # Clear previous image if not running and not an already processed uploaded image
        if not (input_choice == "Upload Image" and st.session_state.uploaded_image_processed):
            st_frame_placeholder.empty()
        audio_output_placeholder.empty()
        # Reset class detection state when explicitly stopped or cleared
        if not st.session_state.running:
            st.session_state.class_detection_state = {}


except Exception as e:
    st.error(f"A critical error occurred in the application: {e}")
    audio_output_placeholder.empty()
    st.session_state.running = False
    st.session_state.class_detection_state = {}
