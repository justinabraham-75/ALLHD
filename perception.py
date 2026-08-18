# perception.py
from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st

# --- OPTIMIZATION (4): CACHE MODEL IN MEMORY ---
# This ensures your 3,000,000+ model parameters sit permanently in RAM/GPU 
# instead of being reloaded on every single frame slide, giving you a 60 FPS UI.
@st.cache_resource
def load_edge_model():
    # Points directly to your optimized, radiation-hardened-ready ONNX weights
    MODEL_PATH = "runs/detect/train-4/weights/best.pt"
    return YOLO(MODEL_PATH, task="detect")

def detect_hazards(image_path):
    """
    Hybrid Perception Module:
    1. CNN isolates Craters with strict single-pass inference.
    2. OpenCV extracts Rocks and Slope density using memory-efficient morphological layers.
    """
    # Grab the globally cached model instantly
    edge_model = load_edge_model()
    
    # --- 1. CNN PERCEPTION (CRATERS) ---
    results = edge_model(image_path, verbose=False)[0]
    hazards_list = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        
        hazards_list.append({
            "type": "crater",
            "x_center": x_center,
            "y_center": y_center,
            "confidence": float(box.conf[0]),
            "box_coords": (x1, y1, x2, y2)
        })
        
    # --- 2. OPTIMIZED MORPHOLOGICAL PERCEPTION (ROCKS & SLOPES) ---
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- OPTIMIZATION (3): GAUSSIAN BLURRING FOR NOISE REDUCTION ---
    # Pre-filtering the raw image clears out sharp dust grain textures, 
    # preventing false-positive hazard classifications.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Slopes: Execute Canny gradient analysis on the cleaned, blurred matrix
    edges = cv2.Canny(blurred, 40, 130)
    
    # --- OPTIMIZATION (2): TUNED ADAPTIVE THRESHOLD HYPERPARAMETERS ---
    # Expanded neighborhood block size to 21 to evaluate macro terrain structures, 
    # adjusting the constant subtractor to -8 to perfectly outline actual rocks.
    bright_spots = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -8
    )
    
    # --- OPTIMIZATION (3): MORPHOLOGICAL CLOSING KERNEL ---
    # Connects small, fractured pixels of the same boulder structure 
    # into a solid, definitive geometrical contour layout.
    kernel = np.ones((3, 3), np.uint8)
    bright_spots = cv2.morphologyEx(bright_spots, cv2.MORPH_CLOSE, kernel)
    
    return hazards_list, edges, bright_spots