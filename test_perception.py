# test_perception.py
import cv2
import numpy as np
from perception import detect_hazards

# Path to one of your unseen lunar images
TEST_IMAGE = "D:\data.jpg" # Update this line

# Run your hybrid pipeline
cnn_hazards, edges, bright_spots = detect_hazards(TEST_IMAGE)

# Load base image to draw diagnostic annotations
display_img = cv2.imread(TEST_IMAGE)

# 1. DRAW CNN CRATERS (RED BOXES)
for hazard in cnn_hazards:
    x1, y1, x2, y2 = hazard["box_coords"]
    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(display_img, f"Crater: {hazard['confidence']:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 2. DRAW OPENCV ROCKS (BLUE OUTLINES)
# Find contours around the bright reflective spot thresholds
contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 5: # Filter micro pixel noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

# 3. DRAW OPENCV SLOPES (YELLOW HIGHLIGHT OVERLAYS)
# If edges are detected, color those specific pixels yellow
display_img[edges > 0] = [0, 255, 255]

# Show the results locally
cv2.imshow("AeroEdge Multi-Hazard Diagnostics", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()