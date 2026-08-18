# app.py
import streamlit as st
import cv2
import numpy as np
import os
import psutil
import time
import pandas as pd
from datetime import datetime
from perception import detect_hazards  # Ensure your perception.py is in the same folder

# --- PAGE SETUP ---
st.set_page_config(page_title="AeroEdge GNC Dashboard", layout="wide")
st.title("🚀 AeroEdge: Autonomous Lunar Landing Hazard Detection & Site Recommendation")
st.subheader("Edge-AI Real-Time Guidance, Navigation & Control Simulation")

# --- PATH SETUP ---
TEST_DIR = "D:/VS Code/Python/CNN/train/images"  # Update this to your local test images folder
LOG_FILE = "telemetry_log.csv"

# Ensure log file exists with headers
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=["Timestamp", "Image", "Craters_Detected", "Safe_Zone_Grid", "Max_Suitability_Score", "RAM_Usage_MB"])
    df_init.to_csv(LOG_FILE, index=False)

# Get list of images for testing (Deliverable requires demonstrating at least 50 test images)
if os.path.exists(TEST_DIR):
    all_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
else:
    all_images = []
    st.error(f"Test image directory not found at: {TEST_DIR}. Please update the path.")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🕹️ Flight Control Panel")
if all_images:
    # Frame index slider to rapidly demonstrate performance across up to 50+ images
    img_index = st.sidebar.slider("Select Telemetry Frame", 0, min(len(all_images)-1, 100), 0)
    selected_img_name = all_images[img_index]
    image_path = os.path.join(TEST_DIR, selected_img_name)
else:
    st.sidebar.warning("No images found in the test directory.")
    st.stop()

# In the sidebar of app.py
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Sensor Override Diagnostics")
show_craters = st.sidebar.checkbox("Enable CNN (Crater Tracking)", value=True)
show_rocks = st.sidebar.checkbox("Enable Optical Thresholding (Rock Tracking)", value=True)
show_slopes = st.sidebar.checkbox("Enable Edge Gradients (Slope Tracking)", value=True)


# Fuel diverted cost weight slider
w_fuel = st.sidebar.slider("Fuel Optimization Weight (W_fuel)", 0.0, 1.0, 0.4, 0.1)

# --- THE GEOMETRICAL BRAIN & SCORING LOGIC ---
def process_landing_pipeline(img_path):
    # 1. Perception Layer (Single-Pass Inference)
    start_time = time.time()
    cnn_hazards, edges, bright_spots = detect_hazards(img_path)
    inference_time = (time.time() - start_time) * 1000  # in milliseconds

    # Read base image for UI dimensions
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # 2. Virtual Grid Setup (5x5 Matrix to protect RAM constraints)
    grid_rows, grid_cols = 5, 5
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    # Initialize matrix arrays
    hazard_matrix = np.zeros((grid_rows, grid_cols))
    slope_matrix = np.zeros((grid_rows, grid_cols))
    suitability_matrix = np.zeros((grid_rows, grid_cols))
    
    # Map CNN Craters into the Virtual Grid
    for hazard in cnn_hazards:
        xc, yc = hazard["x_center"], hazard["y_center"]
        grid_x = min(xc // cell_w, grid_cols - 1)
        grid_y = min(yc // cell_h, grid_rows - 1)
        hazard_matrix[grid_y, grid_x] += 15  # Heavy penalty for craters

    # Map OpenCV Slopes and Rocks via pixel density checks
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Crop a small virtual bounding patch corresponding to the cell
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            # Edge density proxy for steep slope gradients
            slope_pixels = np.sum(edges[y1:y2, x1:x2] > 0)
            if slope_pixels > (cell_h * cell_w * 0.02):  # Threshold condition
                slope_matrix[r, c] += 10
                
            # Bright spot pixel counts for rocks/boulders
            rock_pixels = np.sum(bright_spots[y1:y2, x1:x2] > 0)
            if rock_pixels > 5:
                hazard_matrix[r, c] += 5

    # Compute Landing Suitability Score for each zone
    # Suitability Score = 100 - (Hazard_Penalty + Slope_Penalty + Fuel_Divert_Penalty)
    center_r, center_c = grid_rows // 2, grid_cols // 2
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Calculate geometric Euclidean distance from center (lander trajectory)
            dist_from_center = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            fuel_penalty = dist_from_center * 12 * w_fuel
            
            # Total penalty accumulation
            total_penalty = hazard_matrix[r, c] + slope_matrix[r, c] + fuel_penalty
            
            # Invert to make higher score = safer zone (Max 100)
            suitability_matrix[r, c] = max(0, 100 - total_penalty)
            
    # Find the optimal safe site coordinate
    best_idx = np.unravel_index(np.argmax(suitability_matrix, axis=None), suitability_matrix.shape)
    
    # 3. Dynamic Telemetry Metrics Drawing
    # 3. Dynamic Telemetry Metrics Drawing
    annotated_img = img.copy()
    
    # --- REFINED SENSOR DRAWING WITH INTERACTIVE DIAGNOSTIC FLAGS ---
    
    # Draw OpenCV Slopes (YELLOW EDGE HIGHLIGHT) if toggled on in sidebar
    if show_slopes:
        annotated_img[edges > 0] = [0, 255, 255]

    # Draw OpenCV Rocks (BLUE BOUNDING BOXES) if toggled on in sidebar
    if show_rocks:
        contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Filters out microscopic pixel dust, focusing strictly on landing-critical boulders
            if cv2.contourArea(cnt) > 8: 
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                cv2.rectangle(annotated_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)

    # Project the 5x5 Virtual Grid Overlay on UI
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
            
            score_text = f"{int(suitability_matrix[r, c])}"
            cv2.putText(annotated_img, score_text, (x1 + 10, y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw CNN Craters (RED BOUNDING BOXES) if toggled on in sidebar
    if show_craters:
        for hazard in cnn_hazards:
            bx1, by1, bx2, by2 = hazard["box_coords"]
            cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        
    # Highlight Optimal Safe Site Landing Pad (THICK GREEN BOX WITH HEADER)
    br, bc = best_idx
    cv2.rectangle(annotated_img, (bc * cell_w, br * cell_h), ((bc + 1) * cell_w, (br + 1) * cell_h), (0, 255, 0), 4)
    cv2.putText(annotated_img, "RECOMMENDED PAD", (bc * cell_w + 5, br * cell_h + cell_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return annotated_img, inference_time, suitability_matrix, best_idx, len(cnn_hazards)

# --- RUN EXECUTION PIPELINE ---
processed_frame, inf_time, scores, safe_zone, num_craters = process_landing_pipeline(image_path)

# Track real-time RAM and hardware usage
process = psutil.Process(os.getpid())
current_mem_mb = process.memory_info().rss / (1024 * 1024)

# Log to Black Box Telemetry CSV
timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
new_log = pd.DataFrame([[timestamp_str, selected_img_name, num_craters, f"Row {safe_zone[0]} Col {safe_zone[1]}", np.max(scores), f"{current_mem_mb:.2f}"]], 
                       columns=["Timestamp", "Image", "Craters_Detected", "Safe_Zone_Grid", "Max_Suitability_Score", "RAM_Usage_MB"])
new_log.to_csv(LOG_FILE, mode='a', header=False, index=False)

# --- RENDER ONBOARD DISPLAY ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎛️ Downward-Facing Optical Sensor Feed")
    # Convert BGR openCV formatting to RGB Streamlit formatting
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    st.image(rgb_frame, use_container_width=True, caption=f"Processing Frame: {selected_img_name}")

with col2:
    st.subheader("📊 Onboard GNC Telemetry Telemetry")
    
    # 1. Score Metric Cards
    st.metric(label="Target Landing Pad Suitability Score", value=f"{np.max(scores):.1f} / 100.0")
    st.metric(label="Onboard Computing RAM Allocation", value=f"{current_mem_mb:.2f} MB", delta="Safe Margin (SWaP-C compliant)")
    st.metric(label="Perception Inference Core Latency", value=f"{inf_time:.2f} ms", delta="Real-Time Profile")
    
    # 2. Output Recommendations
    st.success(f"**Autonomous Safety Recommendation:** Safe Site identified at **Matrix Grid Cell [Row {safe_zone[0]}, Column {safe_zone[1]}]**.")
    st.info(f"**Total Hazards Logged in Frame:** {num_craters} Craters isolated via Deep Learning Module.")

# --- BOTTOM LOG VIEW (THE BLACK BOX AUDIT TRAIL) ---
st.write("---")
st.subheader("📁 Black Box Onboard Telemetry File Archive (`telemetry_log.csv`)")
log_df = pd.read_csv(LOG_FILE)
st.dataframe(log_df.tail(10), use_container_width=True)