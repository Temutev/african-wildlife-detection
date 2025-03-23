import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time
import io
import base64
from datetime import datetime
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="African Wildlife Detection",
    page_icon="🦓",
    layout="wide"
)

# Define class names
CLASS_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}

# App title and description
st.title("African Wildlife Detection")
st.markdown("Upload images or videos to detect and track buffalo, elephant, rhino, and zebra in the wild.")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = {}

# Load the model
@st.cache_resource
def load_model(model_path="yolo11n.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
if st.session_state.model is None:
    with st.spinner("Loading YOLO model..."):
        st.session_state.model = load_model()
        if st.session_state.model:
            st.success("Model loaded successfully!")

# Function to perform detection
def perform_detection(image, confidence):
    if st.session_state.model is None:
        st.error("Model not loaded. Please check the model path.")
        return None
    
    start_time = time.time()
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Perform inference
    results = st.session_state.model.predict(
        source=image_np,
        conf=confidence,
        save=False
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    st.session_state.processing_times.append(processing_time)
    
    # Update detection history
    if results and len(results) > 0:
        boxes = results[0].boxes.cpu().numpy() if len(results[0].boxes) > 0 else []
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            st.session_state.detection_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "class": class_name,
                "confidence": float(box.conf[0])
            })
    
    return results, processing_time

# Function to create download link for image
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Simple object tracker class
class SimpleTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
    def update(self, detections):
        # If no existing tracks, create new tracks for all detections
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'hits': 1,
                    'history': [det['bbox']]
                }
                self.next_id += 1
            return self.tracks
            
        # Match detections to existing tracks based on IoU
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        # Calculate IoU between each detection and each track
        for d_idx, det in enumerate(detections):
            for t_idx in unmatched_tracks:
                track = self.tracks[t_idx]
                iou = self._calculate_iou(det['bbox'], track['bbox'])
                if iou > self.iou_threshold:
                    matches.append((d_idx, t_idx, iou))
        
        # Sort matches by IoU (descending)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Update matched tracks
        matched_d_indices = set()
        matched_t_indices = set()
        
        for d_idx, t_idx, _ in matches:
            if d_idx not in matched_d_indices and t_idx not in matched_t_indices:
                # Update the track
                det = detections[d_idx]
                self.tracks[t_idx]['bbox'] = det['bbox']
                self.tracks[t_idx]['confidence'] = det['confidence']
                self.tracks[t_idx]['hits'] += 1
                self.tracks[t_idx]['age'] = 0
                self.tracks[t_idx]['history'].append(det['bbox'])
                
                # Mark as matched
                matched_d_indices.add(d_idx)
                matched_t_indices.add(t_idx)
        
        # Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_d_indices:
                det = detections[d_idx]
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'hits': 1,
                    'history': [det['bbox']]
                }
                self.next_id += 1
        
        # Update unmatched tracks
        tracks_to_delete = []
        for t_idx in self.tracks:
            if t_idx not in matched_t_indices:
                self.tracks[t_idx]['age'] += 1
                
                # Remove tracks that haven't been updated for a while
                if self.tracks[t_idx]['age'] > self.max_age:
                    tracks_to_delete.append(t_idx)
        
        # Delete old tracks
        for t_idx in tracks_to_delete:
            del self.tracks[t_idx]
            
        return self.tracks
    
    def _calculate_iou(self, box1, box2):
        # Extract box coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

# Create a tracker instance
tracker = SimpleTracker()

# Process video with tracking
def process_video(video_path, confidence, frame_skip=5):
    if st.session_state.model is None:
        st.error("Model not loaded. Please check the model path.")
        return None
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary file for output video
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output_file.name
    temp_output_file.close()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process frames
    frame_count = 0
    detection_count = 0
    
    # Reset tracker for new video
    tracker = SimpleTracker()
    tracking_results = {}
    
    # Define colors for different classes (BGRA)
    colors = {
        "buffalo": (255, 0, 0),    # Red
        "elephant": (0, 255, 0),   # Green
        "rhino": (0, 0, 255),      # Blue
        "zebra": (255, 255, 0)     # Cyan
    }
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every N frames to speed up
            if frame_count % frame_skip == 0:
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Perform detection
                results = st.session_state.model.predict(
                    source=frame,
                    conf=confidence,
                    save=False
                )
                
                # Extract detections for tracking
                detections = []
                if results and len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Get coordinates (convert to xyxy format if needed)
                        if hasattr(box, 'xyxy'):
                            x1, y1, x2, y2 = box.xyxy[0]
                        else:
                            x1, y1, x2, y2 = box.xywh[0]  # Convert xywh to xyxy if needed
                            x1, y1 = x1 - x2/2, y1 - y2/2
                            x2, y2 = x1 + x2, y1 + y2
                        
                        class_id = int(box.cls[0])
                        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                        confidence_score = float(box.conf[0])
                        
                        detections.append({
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'class': class_name,
                            'confidence': confidence_score
                        })
                        
                        detection_count += 1
                
                # Update tracker with new detections
                tracks = tracker.update(detections)
                tracking_results[frame_count] = tracks.copy()
                
                # Draw tracking results on frame
                for track_id, track_info in tracks.items():
                    if track_info['hits'] >= tracker.min_hits:
                        x1, y1, x2, y2 = track_info['bbox']
                        class_name = track_info['class']
                        conf = track_info['confidence']
                        
                        # Draw bounding box
                        color = colors.get(class_name, (0, 255, 255))  # Default to yellow if class not found
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw track ID and class name
                        label = f"#{track_id} {class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw trajectory (last N points)
                        if len(track_info['history']) > 1:
                            for i in range(1, min(10, len(track_info['history']))):
                                prev_box = track_info['history'][-i-1]
                                curr_box = track_info['history'][-i]
                                
                                # Get center points of boxes
                                prev_center = (int((prev_box[0] + prev_box[2]) / 2), int((prev_box[1] + prev_box[3]) / 2))
                                curr_center = (int((curr_box[0] + curr_box[2]) / 2), int((curr_box[1] + curr_box[3]) / 2))
                                
                                # Draw line connecting centers
                                cv2.line(frame, prev_center, curr_center, color, 2)
            
            # Write the frame to output video
            out.write(frame)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        # Store tracking data in session state
        st.session_state.tracking_data = tracking_results
            
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None
    finally:
        # Release resources
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
    
    return temp_output_path, detection_count, total_frames, fps

# Sidebar for configuration
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
enable_tracking = st.sidebar.checkbox("Enable Object Tracking", value=True)

# Input options
st.sidebar.header("Input Options")
input_option = st.sidebar.radio(
    "Select Input Method", 
    ["Upload Single Image", "Batch Process Images", "Upload Video", "Capture from Camera", "Use Demo Image"]
)

# Main area for displaying results
if input_option == "Upload Single Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Read image
        image = Image.open(uploaded_file)
        
        # Display original image
        with col1:
            st.header("Original Image")
            st.image(image, use_column_width=True)
            
        # Process image
        with st.spinner("Detecting wildlife..."):
            results, proc_time = perform_detection(image, confidence)
            
        # Display results
        with col2:
            st.header("Detection Results")
            if results and len(results) > 0:
                # Get the plotted result image
                result_img = results[0].plot()
                result_pil = Image.fromarray(result_img)
                st.image(result_img, use_column_width=True)
                
                # Add download button for processed image
                result_filename = f"wildlife_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                st.markdown(get_image_download_link(result_pil, result_filename, "Download Processed Image"), unsafe_allow_html=True)
                
                # Display detection details
                st.subheader("Detected Objects")
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.cpu().numpy()
                    
                    # Create a list to store detection info
                    detections = []
                    
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence_score = float(box.conf[0])
                        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                        
                        detections.append({
                            "Class": class_name,
                            "Confidence": f"{confidence_score:.2f}"
                        })
                    
                    # Display as a table
                    st.table(detections)
                    
                    # Display processing time
                    st.info(f"Processing time: {proc_time:.3f} seconds")
                else:
                    st.info("No wildlife detected in this image.")
            else:
                st.info("No results returned from the model.")

elif input_option == "Batch Process Images":
    uploaded_files = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.header("Batch Processing")
        
        # Add a button to process all images
        if st.button("Process All Images"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Clear previous batch results
            st.session_state.batch_results = []
            
            # Process each image
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Read image
                image = Image.open(uploaded_file)
                
                # Process image
                results, proc_time = perform_detection(image, confidence)
                
                # Store results
                if results and len(results) > 0:
                    result_img = results[0].plot()
                    result_pil = Image.fromarray(result_img)
                    
                    # Create download link
                    result_filename = f"wildlife_{uploaded_file.name}"
                    download_link = get_image_download_link(result_pil, result_filename, "Download")
                    
                    # Get detections
                    detections = []
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes.cpu().numpy()
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence_score = float(box.conf[0])
                            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                            detections.append({"Class": class_name, "Confidence": confidence_score})
                    
                    # Add to batch results
                    st.session_state.batch_results.append({
                        "filename": uploaded_file.name,
                        "image": image,
                        "result_image": result_pil,
                        "detections": detections,
                        "processing_time": proc_time,
                        "download_link": download_link
                    })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Processed {len(uploaded_files)} images successfully!")
        
        # Display batch results
        if st.session_state.batch_results:
            # Create tabs for Summary and Individual Results
            tab1, tab2 = st.tabs(["Summary", "Individual Results"])
            
            with tab1:
                st.subheader("Batch Processing Summary")
                
                # Count total detections by class
                all_detections = []
                for result in st.session_state.batch_results:
                    all_detections.extend([d["Class"] for d in result["detections"]])
                
                detection_counts = Counter(all_detections)
                
                # Create summary statistics
                total_images = len(st.session_state.batch_results)
                total_detections = len(all_detections)
                avg_processing_time = sum(r["processing_time"] for r in st.session_state.batch_results) / total_images if total_images > 0 else 0
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", total_images)
                col2.metric("Total Detections", total_detections)
                col3.metric("Avg. Processing Time", f"{avg_processing_time:.3f}s")
                
                # Create a bar chart of detections by class
                if detection_counts:
                    st.subheader("Detections by Class")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    classes = list(detection_counts.keys())
                    counts = list(detection_counts.values())
                    
                    # Choose colors based on class names
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                    ax.bar(classes, counts, color=colors[:len(classes)])
                    
                    for i, count in enumerate(counts):
                        ax.text(i, count + 0.1, str(count), ha='center')
                    
                    ax.set_ylabel('Count')
                    ax.set_title('Wildlife Detections by Class')
                    
                    st.pyplot(fig)
                    
                    # Create a download link for summary CSV
                    summary_df = pd.DataFrame({
                        "Class": classes,
                        "Count": counts,
                        "Percentage": [count/total_detections*100 if total_detections > 0 else 0 for count in counts]
                    })
                    
                    csv = summary_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="detection_summary.csv">Download Summary CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.info("No detections found in the processed images.")
            
            with tab2:
                st.subheader("Individual Results")
                
                # Create an expander for each processed image
                for i, result in enumerate(st.session_state.batch_results):
                    with st.expander(f"Image {i+1}: {result['filename']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(result["image"], caption="Original Image", use_column_width=True)
                        
                        with col2:
                            st.image(result["result_image"], caption="Processed Image", use_column_width=True)
                            st.markdown(result["download_link"], unsafe_allow_html=True)
                        
                        # Display detection details
                        if result["detections"]:
                            st.dataframe(result["detections"])
                        else:
                            st.info("No wildlife detected in this image.")
                        
                        st.text(f"Processing time: {result['processing_time']:.3f} seconds")

elif input_option == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_path = temp_video_file.name
        temp_video_file.write(uploaded_video.read())
        temp_video_file.close()
        
        st.header("Video Processing")
        
        # Add video processing options
        col1, col2 = st.columns(2)
        with col1:
            frame_skip = st.slider("Process every N frames", 1, 30, 5)
        
        # Process the video
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                output_path, detection_count, total_frames, fps = process_video(
                    temp_video_path, confidence, frame_skip
                )
            
            if output_path:
                st.success(f"Video processed successfully! {detection_count} detections across {total_frames} frames.")
                
                # Display the processed video
                st.video(output_path)
                
                # Provide download link for processed video
                with open(output_path, 'rb') as file:
                    video_bytes = file.read()
                    
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name=f"processed_{uploaded_video.name}",
                    mime="video/mp4"
                )
                
                # Display tracking visualization if enabled
                if enable_tracking and st.session_state.tracking_data:
                    st.subheader("Object Tracking Analysis")
                    
                    # Get all tracked objects
                    all_tracks = set()
                    for frame_id, tracks in st.session_state.tracking_data.items():
                        all_tracks.update(tracks.keys())
                    
                    # Get tracking stats
                    track_stats = []
                    for track_id in all_tracks:
                        # Find frames where this track appears
                        appearances = [frame_id for frame_id, tracks in st.session_state.tracking_data.items() 
                                      if track_id in tracks]
                        
                        if appearances:
                            first_appearance = min(appearances)
                            last_appearance = max(appearances)
                            duration_frames = last_appearance - first_appearance
                            duration_seconds = duration_frames / fps if fps > 0 else 0
                            
                            # Get the class of this track
                            track_class = None
                            for frame_id in appearances:
                                if track_id in st.session_state.tracking_data[frame_id]:
                                    track_class = st.session_state.tracking_data[frame_id][track_id]['class']
                                    break
                            
                            track_stats.append({
                                "Track ID": track_id,
                                "Class": track_class,
                                "Duration (frames)": duration_frames,
                                "Duration (seconds)": f"{duration_seconds:.2f}",
                                "First Frame": first_appearance,
                                "Last Frame": last_appearance
                            })
                    
                    # Display tracking statistics
                    if track_stats:
                        st.dataframe(track_stats)
                        
                        # Count objects by class
                        class_counts = Counter([stat["Class"] for stat in track_stats])
                        
                        # Create a pie chart of tracked objects by class
                        if class_counts:
                            fig, ax = plt.subplots(figsize=(8, 8))
                            labels = list(class_counts.keys())
                            sizes = list(class_counts.values())
                            
                            # Choose colors based on class names
                            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(labels)])
                            ax.axis('equal')
                            ax.set_title('Tracked Wildlife by Class')
                            
                            st.pyplot(fig)
                
                # Clean up temporary files
                os.unlink(temp_video_path)
                os.unlink(output_path)

elif input_option == "Capture from Camera":
    st.sidebar.info("Please allow camera access when prompted.")
    
    # Capture from webcam
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        col1, col2 = st.columns(2)
        
        # Display original image
        with col1:
            st.header("Original Image")
            st.image(camera_image, use_column_width=True)
            
        # Process image
        with st.spinner("Detecting wildlife..."):
            image = Image.open(camera_image)
            results, proc_time = perform_detection(image, confidence)
            
        # Display results
        with col2:
            st.header("Detection Results")
            if results and len(results) > 0:
                # Get the plotted result image
                result_img = results[0].plot()
                result_pil = Image.fromarray(result_img)
                st.image(result_img, use_column_width=True)
                
                # Add download button for processed image
                result_filename = f"wildlife_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                st.markdown(get_image_download_link(result_pil, result_filename, "Download Processed Image"), unsafe_allow_html=True)
                
                # Display detection details
                st.subheader("Detected Objects")
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.cpu().numpy()
                    
                    # Create a list to store detection info
                    detections = []
                    
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence_score = float(box.conf[0])
                        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                        
                        detections.append({
                            "Class": class_name,
                            "Confidence": f"{confidence_score:.2f}"
                        })
                    
                    # Display as a table
                    st.table(detections)
                    
                    # Display processing time
                    st.info(f"Processing time: {proc_time:.3f} seconds")
                else:
                    st.info("No wildlife detected in this image.")
            else:
                st.info("No results returned from the model.")

else:  # Demo Image
    st.sidebar.info("Using a demo image for wildlife detection.")
    
    # In a real app, you would include demo images with your code
    with st.container():
        st.header("Demo Image")
        st.info("In a production app, you would include sample images here.")
        
        # Placeholder for demo functionality
        if st.button("Run Detection on Demo Image"):
            st.info("In a real app, this would show detection results on a demo image.")

# Add Analytics Dashboard
st.header("Model Analytics Dashboard")
tab1, tab2 = st.tabs(["Detection History", "Performance Metrics"])

with tab1:
    if st.session_state.detection_history:
        # Create a dataframe from detection history
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Add filter for class
        selected_class = st.multiselect(
            "Filter by class", 
            options=sorted(df["class"].unique()),
            default=sorted(df["class"].unique())
        )
        
        # Apply filter
        filtered_df = df[df["class"].isin(selected_class)]
        
        # Display detection history
        st.dataframe(filtered_df)
        
        # Show time series of detections
        if not filtered_df.empty:
            st.subheader("Detection Timeline")
            
            # Convert timestamp to datetime
            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"])
            
            # Group by timestamp and class, count detections
            timeline_df = filtered_df.groupby([pd.Grouper(key="timestamp", freq="1min"), "class"]).size().reset_index(name="count")
            
            # Create a pivot table for the line chart
            pivot_df = timeline_df.pivot(index="timestamp", columns="class", values="count").fillna(0)
            
            # Plot the timeline
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot_df.plot(ax=ax)
            ax.set_xlabel("Time")
            ax.set_ylabel("Number of Detections")
            ax.set_title("Wildlife Detections Over Time")
            ax.legend(title="Species")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    else:
        st.info("No detection history available. Process some images or videos to see analytics.")

with tab2:
    # Display performance metrics
    if st.session_state.processing_times:
        st.subheader("Processing Time Statistics")
        
        # Calculate statistics
        avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
        max_time = max(st.session_state.processing_times)
        min_time = min(st.session_state.processing_times)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Processing Time", f"{avg_time:.3f}s")
        col2.metric("Max Processing Time", f"{max_time:.3f}s")
        col3.metric("Min Processing Time", f"{min_time:.3f}s")
        
        # Plot histogram of processing times
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(st.session_state.processing_times, bins=10, alpha=0.7, color='blue')
        ax.set_xlabel("Processing Time (s)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Image Processing Times")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    else:
        st.info("No performance metrics available. Process some images or videos to see statistics.")

# Display model information
st.sidebar.header("Model Information")
st.sidebar.info("""
This model was trained to detect:
- Buffalo
- Elephant
- Rhino
- Zebra

Model: YOLOv11 trained on African Wildlife Dataset
""")

# Add instructions and about section
with st.expander("How to use this app"):
    st.markdown("""
    ## Basic Usage
    1. Select your input method from the sidebar:
       - Upload Single Image: Process one image at a time
       - Batch Process Images: Upload and process multiple images
       - Upload Video: Process video files
       - Capture from Camera: Use your webcam
       - Use Demo Image: Try with example images
    
    2. Adjust the confidence threshold to control detection sensitivity
    
    3. Enable object tracking for videos to follow animal movements
    
    ## Features
    - **Download Results**: Save processed images with detection boxes
    - **Batch Processing**: Process multiple images and get summary statistics
    - **Video Processing**: Analyze videos with object tracking
    - **Analytics Dashboard**: View detection history and performance metrics
    """)

with st.expander("About"):
    st.markdown("""
    This application uses a YOLOv11 model trained on the African Wildlife Dataset to detect four types of animals commonly found in African wildlife reserves.
    
    The model was trained for 100 epochs on images of buffalo, elephants, rhinos, and zebras.
    
    ### Technologies Used
    - Streamlit: Web application framework
    - Ultralytics YOLOv11: Object detection model
    - OpenCV: Image and video processing
    - Matplotlib & Pandas: Data visualization and analysis
    """)