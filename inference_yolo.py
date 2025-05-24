import streamlit as st
from ultralytics import YOLO
import cv2
import time
import os
import base64
from PIL import Image
import numpy as np
import io
# streamlit-webrtc and av are no longer needed
# import threading # No longer needed

# --- Configuration ---
MODEL_PATH = "model/best.pt"  # Please ensure this path is correct
AUDIO_DIR = "."  # Directory where your .mp3 files are located
# DETECTION_THRESHOLD_FOR_PLAY is not directly applicable for snapshots
DETECTION_COOLDOWN_AFTER_PLAY = 5.0  # Seconds to wait before playing audio for the same class again
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

# --- Streamlit App Layout ---
st.title("Money Detection using YOLO")

# Initialize session state variables if they don't exist
if 'uploaded_image_processed' not in st.session_state:
    st.session_state.uploaded_image_processed = False
if 'class_detection_state' not in st.session_state: # Still useful for cooldown
    st.session_state.class_detection_state = {}
if 'confidence_threshold_snapshot' not in st.session_state:
    st.session_state.confidence_threshold_snapshot = DEFAULT_CONFIDENCE_THRESHOLD
if 'confidence_threshold_upload' not in st.session_state:
    st.session_state.confidence_threshold_upload = DEFAULT_CONFIDENCE_THRESHOLD

# --- Input Choice ---
input_choice = st.radio("Choose input source:", ("Take Snapshot", "Upload Image")) # Changed "Webcam" to "Take Snapshot"

# --- Placeholders ---
audio_output_placeholder = st.empty()
st_frame_placeholder = st.empty() # For displaying images

# --- Main Processing Logic ---
if input_choice == "Take Snapshot":
    st.session_state.uploaded_image_processed = False # Reset upload state
    # No need for st.session_state.running for snapshot mode

    confidence_snapshot_slider = st.slider(
        "Confidence Threshold (Snapshot)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.confidence_threshold_snapshot,
        step=0.01,
        help="Adjust for snapshot detection.",
        key="snapshot_conf_slider"
    )
    st.session_state.confidence_threshold_snapshot = confidence_snapshot_slider

    # Clear previous image/message from placeholder if any
    # st_frame_placeholder.empty() # Clear any previous image from upload mode

    img_file_buffer = st.camera_input("Take a picture using your webcam:")

    if img_file_buffer is not None:
        st_frame_placeholder.empty() # Clear placeholder before showing new image
        audio_output_placeholder.empty() # Clear previous audio

        st.write("Processing snapshot...")
        try:
            # To read image file buffer as bytes:
            bytes_data = img_file_buffer.getvalue()
            # Convert bytes data to PIL Image
            pil_image = Image.open(io.BytesIO(bytes_data))

            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')

            # Convert PIL Image to OpenCV image (NumPy array)
            frame_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            annotated_frame_cv2 = frame_cv2.copy()
            current_time = time.time()

            results = model(frame_cv2, conf=st.session_state.confidence_threshold_snapshot, verbose=False)
            
            detected_classes_in_snapshot = set()
            played_audio_for_snapshot_run = set() # To ensure one sound per class name in this specific snapshot run

            if results and results[0].boxes:
                result = results[0]
                annotated_frame_cv2 = result.plot(img=annotated_frame_cv2, conf=True)
                boxes = result.boxes

                for i in range(len(boxes)):
                    try:
                        class_id = int(boxes.cls[i].item())
                        class_name = model.names[class_id]
                        detected_classes_in_snapshot.add(class_name)

                        # Initialize state for new class if not present
                        if class_name not in st.session_state.class_detection_state:
                            st.session_state.class_detection_state[class_name] = {'last_played_time': 0}
                        
                        class_state = st.session_state.class_detection_state[class_name]
                        time_since_last_played_for_class = current_time - class_state['last_played_time']

                        if (time_since_last_played_for_class >= DETECTION_COOLDOWN_AFTER_PLAY and
                            class_name not in played_audio_for_snapshot_run):
                            
                            audio_file_name = f"{class_name}.mp3"
                            audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                            if os.path.exists(audio_file_path):
                                autoplay_audio_html(audio_file_path, audio_output_placeholder)
                                class_state['last_played_time'] = current_time
                                played_audio_for_snapshot_run.add(class_name)
                                # If multiple distinct sounds, this might clear previous before it finishes.
                                # For snapshot, maybe collect all sounds and play them sequentially or just one.
                                # For now, it will attempt to play each, one after another if cooldown allows.
                                time.sleep(0.2) # Small delay if multiple sounds are triggered
                            else:
                                print(f"Audio file not found for '{class_name}': {audio_file_path}")

                    except Exception as e:
                        print(f"Error processing a detection or audio for snapshot: {e}")
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_cv2, cv2.COLOR_BGR2RGB)
            st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Processed Snapshot", use_container_width=True)
            st.success("Snapshot processed.")

        except Exception as e:
            st.error(f"An error occurred during snapshot processing: {e}")
            st_frame_placeholder.empty()
    elif not img_file_buffer: # If camera_input is present but no image taken yet, or cleared.
        st_frame_placeholder.info("Use the button above to take a snapshot.")
        audio_output_placeholder.empty()


elif input_choice == "Upload Image":
    # Clear snapshot specific things if any
    # (snapshot mode already clears placeholders when it runs)
    audio_output_placeholder.empty() 

    confidence_upload_slider = st.slider(
        "Confidence Threshold (Image Upload)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.confidence_threshold_upload,
        step=0.01,
        help="Adjust for image upload detection.",
        key="upload_conf_slider"
    )
    st.session_state.confidence_threshold_upload = confidence_upload_slider

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    if st.button("Process Uploaded Image", key="start_image_processing"):
        if uploaded_file is not None:
            st.session_state.uploaded_image_processed = False
            st_frame_placeholder.empty()
            audio_output_placeholder.empty()
            # Reset class_detection_state or use a local set for cooldowns within this image process
            # For simplicity, image upload audio plays if file exists, cooldown logic might be less critical here or handled differently.
            # Let's use a simple set to avoid replaying for the same class in *this single image*.

            st.write("Processing uploaded image...")
            try:
                image_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                image_np = np.array(pil_image)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                annotated_frame = frame.copy()
                
                played_audio_for_image_upload = set()

                results = model(frame, conf=st.session_state.confidence_threshold_upload, verbose=False)
                if results and results[0].boxes:
                    result = results[0]
                    annotated_frame = result.plot(img=annotated_frame, conf=True)
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        try:
                            class_id = int(boxes.cls[i].item())
                            class_name = model.names[class_id]

                            if class_name not in played_audio_for_image_upload:
                                audio_file_name = f"{class_name}.mp3"
                                audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                                if os.path.exists(audio_file_path):
                                    autoplay_audio_html(audio_file_path, audio_output_placeholder)
                                    played_audio_for_image_upload.add(class_name)
                                    time.sleep(0.5) 
                                else:
                                    print(f"Audio file not found for '{class_name}': {audio_file_path}")
                        except Exception as e:
                            print(f"Error processing a detection box or audio for uploaded image: {e}")
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Processed Image", use_container_width=True)
                st.session_state.uploaded_image_processed = True
                st.success("Image processed.")

            except Exception as e:
                st.error(f"An error occurred during image processing: {e}")
                st_frame_placeholder.empty()
        
        elif uploaded_file is None:
            st.warning("Please upload an image first before clicking 'Process Uploaded Image'.")
    
    if not st.session_state.uploaded_image_processed and uploaded_file is None:
         st_frame_placeholder.info("Upload an image and click 'Process Uploaded Image'.")
