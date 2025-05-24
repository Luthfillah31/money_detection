import streamlit as st
from ultralytics import YOLO # Assuming ultralytics YOLO
import cv2
import time
import os
import base64
from PIL import Image
import numpy as np
import io

# --- Configuration ---
MODEL_PATH = "models/best.pt" # Please ensure this path is correct
MODEL_PATH = "model/best.pt" # Please ensure this path is correct
AUDIO_DIR = "." # Directory where your .mp3 files are located
DETECTION_THRESHOLD_FOR_PLAY = 2.0 # Seconds an object must be detected before playing audio
DETECTION_COOLDOWN_AFTER_PLAY = 5.0 # Seconds to wait before playing audio for the same class again
WEBCAM_ID = 0 # 0 for default webcam, change if you have multiple
DEFAULT_CONFIDENCE_THRESHOLD = 0.25 # Default confidence threshold for YOLO model

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
        # Attempt to load the YOLO model.
        # Ensure the YOLO class is correctly defined/imported if not using ultralytics
        model = YOLO(path) 

        success_message_placeholder.success(
            f"YOLO model loaded successfully.\nModel path: {path}"
        )

        # Pause execution for a few seconds.
        time.sleep(3) # Keep success message for 3 seconds

        # Clear the success messages.
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
        # st.warning(f"Audio file not found: {file_path}") # This can be noisy, using print instead
        print(f"Audio file not found: {file_path}")
        return

    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            # Autoplay is not always reliable due to browser restrictions.
            # Users might need to interact with the page first.
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

# Stop the app if the model failed to load
if model is None:
    st.error("Application cannot start as the YOLO model failed to load. Please check the path and model file.")
    st.stop()

# --- Streamlit App Layout ---
st.title("Money Detection using YOLO")

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
if 'running' not in st.session_state:
    st.session_state.running = False
if 'uploaded_image_processed' not in st.session_state:
    st.session_state.uploaded_image_processed = False

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Detection", key="start_detection")
with col2:
    stop_button = st.button("Stop Detection", key="stop_detection")

if start_button:
    st.session_state.running = True
    st.session_state.uploaded_image_processed = False # Reset for new detection
elif stop_button:
    st.session_state.running = False
    # st.session_state.uploaded_image_processed = False # Reset for new detection - might not be needed here

# --- State for tracking detection times and played status ---
if 'class_detection_state' not in st.session_state:
    st.session_state.class_detection_state = {}

# --- Placeholders ---
audio_output_placeholder = st.empty() # For embedding audio
st_frame_placeholder = st.empty() # For displaying video frames or images

# --- Main Processing Logic ---
cap = None # Initialize webcam capture object

try:
    if input_choice == "Webcam":
        if st.session_state.running:
            st.write("Attempting to start webcam feed...")
            try:
                cap = cv2.VideoCapture(WEBCAM_ID)
                if not cap.isOpened():
                    st.error(f"Error: Could not open webcam with ID {WEBCAM_ID}.")
                    st.warning("Please check if a webcam is connected, not in use by another application, and the ID is correct.")
                    st.session_state.running = False # Stop if webcam fails
                else:
                    st.success("Webcam started. Displaying feed...")
                    while st.session_state.running: # Main loop for webcam processing
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Warning: Cannot receive frame (stream end or webcam error?). Exiting loop.")
                            st.session_state.running = False # Stop if no frame
                            break

                        current_time = time.time()
                        # Perform inference with the selected confidence threshold
                        results = model(frame, conf=confidence_slider, verbose=False) 
                        annotated_frame = frame.copy() # Make a copy to draw on

                        detected_classes_in_current_frame = set()

                        if results and results[0].boxes: # Check if there are any detections
                            result = results[0] # Get the first result object
                            # Plot detections on the frame (this is an ultralytics specific method)
                            annotated_frame = result.plot(img=annotated_frame, conf=True) # conf=True shows confidence on plot
                            boxes = result.boxes # Get bounding box information

                            for i in range(len(boxes)):
                                try:
                                    class_id = int(boxes.cls[i].item())
                                    class_name = model.names[class_id] # Get class name from model
                                    detected_classes_in_current_frame.add(class_name)

                                    # Initialize state for new class
                                    if class_name not in st.session_state.class_detection_state:
                                        st.session_state.class_detection_state[class_name] = {
                                            'first_detected_time': current_time,
                                            'last_played_time': 0 # Initialize to 0 to allow immediate play if conditions met
                                        }

                                    class_state = st.session_state.class_detection_state[class_name]
                                    time_since_first_detection = current_time - class_state['first_detected_time']
                                    time_since_last_played = current_time - class_state['last_played_time']

                                    # Check conditions for playing audio
                                    if (time_since_first_detection >= DETECTION_THRESHOLD_FOR_PLAY and
                                        time_since_last_played >= DETECTION_COOLDOWN_AFTER_PLAY):

                                        audio_file_name = f"{class_name}.mp3" # Assumes audio files are named like 'classname.mp3'
                                        audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)

                                        if os.path.exists(audio_file_path):
                                            audio_output_placeholder.empty() # Clear previous audio
                                            autoplay_audio_html(audio_file_path, audio_output_placeholder)
                                            class_state['last_played_time'] = current_time # Update last played time
                                        else:
                                            # This warning can be frequent if audio files are missing
                                            print(f"Audio file not found for '{class_name}': {audio_file_path}")
                                            # st.sidebar.warning(f"Audio file missing: {audio_file_name}") # Option for less intrusive warning
                                except Exception as e:
                                    print(f"Error processing a detection box or audio logic: {e}")

                        # Reset 'first_detected_time' for classes no longer in view
                        # This makes the DETECTION_THRESHOLD_FOR_PLAY reset if an object disappears and reappears
                        classes_to_reset_timer = []
                        for class_name_in_state in list(st.session_state.class_detection_state.keys()): # Iterate over a copy of keys
                            if class_name_in_state not in detected_classes_in_current_frame:
                                # Option 1: Reset timer completely
                                st.session_state.class_detection_state[class_name_in_state]['first_detected_time'] = current_time
                                # Option 2: Or remove the class from state if you want it to be "forgotten"
                                # del st.session_state.class_detection_state[class_name_in_state]

                        # Display the annotated frame
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

                        time.sleep(0.01) # Small delay to control frame rate and allow UI to update

            except cv2.error as e:
                st.error(f"OpenCV Error: {e}. This might be an issue with webcam access or image processing.")
                st.session_state.running = False
            except Exception as e:
                st.error(f"An error occurred during webcam processing: {e}")
                st.session_state.running = False
            finally:
                if cap is not None and cap.isOpened():
                    cap.release()
                    st.write("Webcam released.")
                # Clear audio and detection state when webcam stops or on error
                audio_output_placeholder.empty() 
                st.session_state.class_detection_state = {}

        elif not st.session_state.running and not st.session_state.uploaded_image_processed:
            st_frame_placeholder.empty() # Clear previous image/feed
            st.info("Detection is stopped. Click 'Start Detection' to begin webcam feed.")
            audio_output_placeholder.empty()
            st.session_state.class_detection_state = {}

    elif input_choice == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

        if uploaded_file is not None and st.session_state.running and not st.session_state.uploaded_image_processed:
            st.write("Processing uploaded image...")
            try:
                image_bytes = uploaded_file.read()
                # Convert image bytes to PIL Image, then to NumPy array
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Ensure image is in RGB format if it has an alpha channel (e.g., PNG)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')

                image_np = np.array(pil_image)

                # Convert RGB (from PIL) to BGR (for OpenCV)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                current_time = time.time() # For consistency with webcam logic, though less critical for single image
                # Perform inference with the selected confidence threshold
                results = model(frame, conf=confidence_slider, verbose=False)
                annotated_frame = frame.copy()

                detected_classes_in_current_frame = set() # Keep track of detections in the image

                if results and results[0].boxes:
                    result = results[0]
                    annotated_frame = result.plot(img=annotated_frame, conf=True)
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        try:
                            class_id = int(boxes.cls[i].item())
                            class_name = model.names[class_id]
                            detected_classes_in_current_frame.add(class_name)

                            # For image upload, we can play audio once per detected class if desired
                            # The timing logic (DETECTION_THRESHOLD_FOR_PLAY) might be less relevant
                            # For simplicity, play if detected and cooldown allows

                            if class_name not in st.session_state.class_detection_state:
                                st.session_state.class_detection_state[class_name] = {
                                    'first_detected_time': current_time, # Less relevant for single image
                                    'last_played_time': 0
                                }

                            class_state = st.session_state.class_detection_state[class_name]
                            time_since_last_played = current_time - class_state['last_played_time']

                            # Play audio if cooldown period has passed (or if first time)
                            if time_since_last_played >= DETECTION_COOLDOWN_AFTER_PLAY: # Or simply play if detected
                                audio_file_name = f"{class_name}.mp3"
                                audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                                if os.path.exists(audio_file_path):
                                    # For single image, clear and play for each relevant class
                                    # This might lead to overlapping audio if multiple classes trigger.
                                    # Consider collecting all classes and then deciding on audio playback strategy.
                                    temp_audio_placeholder = st.empty() # Use a temporary placeholder for each sound
                                    autoplay_audio_html(audio_file_path, temp_audio_placeholder)
                                    class_state['last_played_time'] = current_time
                                    # time.sleep(1) # Optional small delay if sounds overlap too much
                                else:
                                    print(f"Audio file not found for '{class_name}': {audio_file_path}")
                        except Exception as e:
                            print(f"Error processing a detection box or audio logic for uploaded image: {e}")

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Processed Image", use_container_width=True)

                st.session_state.uploaded_image_processed = True
                st.session_state.running = False # Stop detection automatically after processing one image

            except Exception as e:
                st.error(f"An error occurred during image processing: {e}")
                st.session_state.running = False # Ensure running is false on error
            finally:
                # For image upload, audio_output_placeholder might not be the best for multiple sounds.
                # The temporary placeholder logic above is one way.
                # Resetting global state after one image makes sense.
                audio_output_placeholder.empty() 
                st.session_state.class_detection_state = {} # Reset state after processing

        elif uploaded_file is None and st.session_state.running:
            st.warning("Please upload an image to start detection, or click 'Stop Detection'.")
            # st.session_state.running = False # Optionally stop if no file is chosen after start
        elif not st.session_state.running and not st.session_state.uploaded_image_processed:
            st_frame_placeholder.empty()
            st.info("Upload an image and click 'Start Detection' to process it.")
            audio_output_placeholder.empty()
            st.session_state.class_detection_state = {}
        elif st.session_state.uploaded_image_processed and not st.session_state.running:
             st.info("Image processed. Upload a new one or switch to webcam.")


except Exception as e:
    st.error(f"A critical error occurred in the application: {e}")
    # Clean up resources if possible
    if cap is not None and cap.isOpened():
        cap.release()
    audio_output_placeholder.empty()
    st.session_state.running = False
    st.session_state.class_detection_state = {}
