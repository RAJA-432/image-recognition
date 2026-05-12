#!/usr/bin/env python3
"""
Quick demo - run YOLOv8 inference without training
Uses pre-trained COCO model directly
"""
import cv2
import numpy as np
from ultralytics import YOLO

# Global state
running = True

def on_key_press(key):
    """Handle keyboard input for stop function"""
    global running
    if key & 0xFF == ord('q') or key & 0xFF == ord('x') or key & 0xFF == 27:  # q, x, or ESC
        running = False

def draw_pointer(frame, center, color=(0, 255, 0)):
    """Draw pointer/crosshair at center of frame"""
    cx, cy = int(center[0]), int(center[1])
    size = 15
    thickness = 2

    # Crosshair
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(frame, (cx, cy), 3, color, -1)

    return frame

def main():
    global running

    # Load pre-trained YOLOv8n model (downloads automatically)
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam...")
    print("Controls: 'q' or 'x' or 'ESC' to quit")

    # Get frame center for pointer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center = (frame_width // 2, frame_height // 2)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=0.25, iou=0.45, show=False)

        # Get detection results
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            class_ids = boxes.cls.int().tolist()
            confidences = boxes.conf.tolist()
            names = model.names

            # Draw bounding boxes with labels
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = class_ids[i]
                conf = confidences[i]
                label = f"{names[cls_id]} {conf:.2f}"

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 255, 0), -1)

                # Draw text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw pointer at center of frame
            frame = draw_pointer(frame, center, color=(255, 0, 0))

        # Display status
        cv2.putText(frame, "Press 'q' or 'x' or 'ESC' to Stop", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('YOLOv8 Object Detection', frame)

        # Check for key press with stop function
        key = cv2.waitKey(1)
        on_key_press(key)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped - resources released")

if __name__ == '__main__':
    main()
