import streamlit as st
from ultralytics import YOLO
import cv2
import time
import os
import base64
from PIL import Image
import numpy as np
import io
# Remove ClientSettings from this import
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# --- Configuration ---
MODEL_PATH = "model/best.pt"  # Please ensure this path is correct
AUDIO_DIR = "."  # Directory where your .mp3 files are located
DETECTION_THRESHOLD_FOR_PLAY = 2.0 # Seconds an object must be detected before playing audio
DETECTION_COOLDOWN_AFTER_PLAY = 5.0  # Seconds to wait before playing audio for the same class again
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
        time.sleep(1)
        success_message_placeholder.empty()
        return model
    except FileNotFoundError:
        success_message_placeholder.empty()
        st.error(f"Error loading YOLO model: Model file not found at {path}")
        return None
    except ImportError:
        success_message_placeholder.empty()
        st.error("Error loading YOLO model: 'ultralytics' library might not be installed.")
        return None
    except Exception as e:
        success_message_placeholder.empty()
        st.error(f"An unexpected error occurred while loading YOLO model from {path}: {e}")
        return None

# --- Autoplay Audio Function ---
def play_audio_from_session_state(audio_placeholder):
    if "play_audio_file_path" in st.session_state and st.session_state.play_audio_file_path:
        file_path = st.session_state.play_audio_file_path
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            st.session_state.play_audio_file_path = None
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
                print(f"Attempted to play audio (from session_state): {file_path}")
        except Exception as e:
            print(f"Error embedding or playing audio {file_path}: {e}")
        finally:
            st.session_state.play_audio_file_path = None


# --- Load Model ---
model = load_yolo_model(MODEL_PATH)
if model is None:
    st.error("Application cannot start as the YOLO model failed to load.")
    st.stop()

# --- Video Transformer Class for streamlit-webrtc ---
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.yolo_model = model
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.class_detection_state_internal = {}

    def update_confidence_threshold(self, new_threshold):
        self.confidence_threshold = new_threshold
    
    def reset_detection_state(self):
        self.class_detection_state_internal = {}

    def recv(self, frame: np.ndarray) -> np.ndarray:
        if not st.session_state.get('running', False):
            return frame

        frm = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        results = self.yolo_model(frm, conf=self.confidence_threshold, verbose=False)
        annotated_frame = frm.copy()
        detected_classes_in_current_frame = set()

        if results and results[0].boxes:
            result = results[0]
            annotated_frame = result.plot(img=annotated_frame, conf=True)
            boxes = result.boxes

            for i in range(len(boxes)):
                try:
                    class_id = int(boxes.cls[i].item())
                    class_name = self.yolo_model.names[class_id]
                    detected_classes_in_current_frame.add(class_name)

                    if class_name not in self.class_detection_state_internal:
                        self.class_detection_state_internal[class_name] = {
                            'first_detected_time': current_time,
                            'last_played_time': 0
                        }
                    
                    class_state = self.class_detection_state_internal[class_name]
                    if class_state.get('first_detected_time', current_time) > current_time - 0.1:
                         class_state['first_detected_time'] = current_time

                    time_since_first_detection = current_time - class_state['first_detected_time']
                    time_since_last_played = current_time - class_state['last_played_time']

                    if (time_since_first_detection >= DETECTION_THRESHOLD_FOR_PLAY and
                        time_since_last_played >= DETECTION_COOLDOWN_AFTER_PLAY):
                        
                        audio_file_name = f"{class_name}.mp3"
                        audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                        
                        if os.path.exists(audio_file_path) and not st.session_state.get("play_audio_file_path"):
                            st.session_state.play_audio_file_path = audio_file_path
                            class_state['last_played_time'] = current_time 
                        elif not os.path.exists(audio_file_path):
                             print(f"Audio file not found for '{class_name}': {audio_file_path}")
                except Exception as e:
                    print(f"Error processing a detection or audio logic in transformer: {e}")
        
        for class_name_in_state in list(self.class_detection_state_internal.keys()):
            if class_name_in_state not in detected_classes_in_current_frame:
                self.class_detection_state_internal[class_name_in_state]['first_detected_time'] = current_time 
        return annotated_frame

if 'video_transformer' not in st.session_state:
    st.session_state.video_transformer = YOLOVideoTransformer()

st.title("Money Detection using YOLO 💰 (Real-time Webcam)")
input_choice = st.radio("Choose input source:", ("Webcam", "Upload Image"))

confidence_slider_val = st.slider(
    "Confidence Threshold",
    min_value=0.0, max_value=1.0,
    value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.01,
    help="Adjust the minimum confidence for object detection.",
    key="confidence_slider_main"
)
if st.session_state.video_transformer.confidence_threshold != confidence_slider_val:
    st.session_state.video_transformer.update_confidence_threshold(confidence_slider_val)

if 'running' not in st.session_state: st.session_state.running = False
if 'uploaded_image_processed' not in st.session_state: st.session_state.uploaded_image_processed = False
if 'play_audio_file_path' not in st.session_state: st.session_state.play_audio_file_path = None

col1, col2 = st.columns(2)
with col1:
    if st.button("Start/Process", key="start_processing"):
        st.session_state.running = True
        st.session_state.uploaded_image_processed = False
        st.session_state.play_audio_file_path = None
        st.session_state.video_transformer.reset_detection_state()
with col2:
    if st.button("Stop/Clear", key="stop_clear"):
        st.session_state.running = False
        st.session_state.uploaded_image_processed = False
        st.session_state.play_audio_file_path = None
        st.session_state.video_transformer.reset_detection_state()

audio_output_placeholder = st.empty()
st_frame_placeholder = st.empty()

try:
    if input_choice == "Webcam":
        if st.session_state.running:
            st.info("Webcam detection is active. Processing frames...")
            webrtc_ctx = webrtc_streamer(
                key="yolo_webcam",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=lambda: st.session_state.video_transformer,
                async_processing=True,
                # ClientSettings removed from here
                media_stream_constraints={ # Still good to keep media_stream_constraints
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
            )
            if not webrtc_ctx.state.playing:
                st.session_state.video_transformer.reset_detection_state()
        else:
            st_frame_placeholder.empty()
            st.info("Webcam detection is stopped. Click 'Start/Process' to begin.")
            st.session_state.video_transformer.reset_detection_state()

    elif input_choice == "Upload Image":
        st.info("Upload an image, then press 'Start/Process'.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

        if uploaded_file is not None and st.session_state.running and not st.session_state.uploaded_image_processed:
            st.write("Processing uploaded image...")
            try:
                image_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode == 'RGBA': pil_image = pil_image.convert('RGB')
                image_np = np.array(pil_image)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                current_time = time.time()
                results = model(frame, conf=confidence_slider_val, verbose=False)
                annotated_frame = frame.copy()
                if results and results[0].boxes:
                    annotated_frame = results[0].plot(img=annotated_frame, conf=True)
                    for i in range(len(results[0].boxes)):
                        class_id = int(results[0].boxes.cls[i].item())
                        class_name = model.names[class_id]
                        audio_file_name = f"{class_name}.mp3"
                        audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
                        if os.path.exists(audio_file_path) and not st.session_state.get("play_audio_file_path"):
                            if 'single_img_audio_played' not in st.session_state: st.session_state.single_img_audio_played = {}
                            if current_time - st.session_state.single_img_audio_played.get(class_name, 0) > DETECTION_COOLDOWN_AFTER_PLAY:
                                st.session_state.play_audio_file_path = audio_file_path
                                st.session_state.single_img_audio_played[class_name] = current_time
                                break 
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Processed Image", use_column_width=True)
                st.session_state.uploaded_image_processed = True
                st.success("Uploaded image processed!")
            except Exception as e:
                st.error(f"An error occurred during image processing: {e}")
                st.session_state.running = False
        
        elif not st.session_state.running and not st.session_state.uploaded_image_processed:
            st_frame_placeholder.empty(); st.info("Upload an image and click 'Start/Process' to process it.")
        elif st.session_state.uploaded_image_processed and not st.session_state.running:
            st.info("Image processed. Upload a new one or switch to webcam.")

    play_audio_from_session_state(audio_output_placeholder)

    if not st.session_state.running:
        if not (input_choice == "Upload Image" and st.session_state.uploaded_image_processed):
            if input_choice == "Webcam": pass 
            else: st_frame_placeholder.empty()
        if st.session_state.play_audio_file_path is None: audio_output_placeholder.empty()
except Exception as e:
    st.error(f"A critical error occurred in the application: {e}")
    st.session_state.running = False
    st.session_state.play_audio_file_path = None
    audio_output_placeholder.empty()
